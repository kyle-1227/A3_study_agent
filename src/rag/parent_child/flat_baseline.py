"""Strict, explicitly addressed flat-baseline artifacts for benchmark comparison.

This module is intentionally separate from the legacy runtime retriever.  It
preserves its Vector -> BM25 -> content-dedup -> reranker ordering, while
requiring explicit resources and exact cleaned-source provenance so a benchmark
can compare it with a parent-child generation without consulting active state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter_ns
from typing import Literal, Protocol

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rank_bm25 import BM25Okapi

from src.rag.parent_child._storage_io import (
    canonical_json_bytes,
    sha256_path,
    sha256_bytes,
)
from src.rag.parent_child.bm25_artifact import digest_identifier_set
from src.rag.parent_child.chroma_runtime_snapshot import (
    CHROMA_RUNTIME_OWNER_SCHEMA_VERSION,
    ChromaRuntimeSnapshot,
)
from src.rag.parent_child.embedding_batches import (
    EmbeddingBatchExecutionError,
    iter_bounded_document_embedding_batches,
)
from src.rag.parent_child.manifests import EmbeddingManifestIdentity
from src.rag.parent_child.retrieval import RerankCandidate, RerankScore


_SHA1_PATTERN = r"^[0-9a-f]{40}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_FLAT_CHUNK_ID_PATTERN = r"^flat_[0-9a-f]{40}$"
_DOC_ID_PATTERN = r"^doc_[0-9a-f]{40}$"


class FlatBaselineError(RuntimeError):
    """A flat-baseline artifact or required retrieval channel is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def make_flat_chunk_id(
    *,
    doc_id: str,
    policy_id: str,
    start_char: int,
    end_char: int,
    content_sha1: str,
) -> str:
    """Return an unambiguous flat chunk identifier derived from exact span data."""

    if start_char < 0 or end_char <= start_char:
        raise ValueError("flat chunk offsets must identify a non-empty span")
    payload = [
        "flat_chunk_id_v1",
        doc_id,
        policy_id,
        start_char,
        end_char,
        content_sha1,
    ]
    return "flat_" + hashlib.sha1(canonical_json_bytes(payload)).hexdigest()


class FlatBaselineChunkMetadata(_StrictFrozenModel):
    """Scalar-safe, policy-independent provenance for one flat baseline chunk."""

    schema_version: Literal["flat_baseline_chunk_metadata_v1"]
    chunk_id: str = Field(pattern=_FLAT_CHUNK_ID_PATTERN)
    doc_id: str = Field(pattern=_DOC_ID_PATTERN)
    subject: str = Field(min_length=1)
    policy_id: str = Field(pattern=_SHA256_PATTERN)
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    chunk_chars: int = Field(gt=0)
    content_sha1: str = Field(pattern=_SHA1_PATTERN)
    source_file: str = Field(min_length=1)
    source_relpath: str = Field(min_length=1)
    source_file_sha1: str = Field(pattern=_SHA1_PATTERN)
    doc_type: str = Field(min_length=1)
    section_path: tuple[str, ...]
    pagination_kind: Literal["physical", "logical"]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    @field_validator("section_path")
    @classmethod
    def _validate_section_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or item != item.strip() for item in value):
            raise ValueError("section_path items must be non-empty and stripped")
        return value

    @model_validator(mode="after")
    def _validate_contract(self) -> FlatBaselineChunkMetadata:
        if self.end_char <= self.start_char:
            raise ValueError("flat chunk span must be non-empty")
        if self.chunk_chars != self.end_char - self.start_char:
            raise ValueError("chunk_chars must equal the exact cleaned span length")
        if self.page_end < self.page_start:
            raise ValueError("page range must be ordered")
        if (
            Path(self.source_relpath).is_absolute()
            or ".." in Path(self.source_relpath).parts
        ):
            raise ValueError("source_relpath must be contained and relative")
        if Path(self.source_relpath).name != self.source_file:
            raise ValueError("source_file must match source_relpath basename")
        expected = make_flat_chunk_id(
            doc_id=self.doc_id,
            policy_id=self.policy_id,
            start_char=self.start_char,
            end_char=self.end_char,
            content_sha1=self.content_sha1,
        )
        if self.chunk_id != expected:
            raise ValueError("flat chunk ID does not match exact provenance")
        return self

    def to_chroma_metadata(self) -> dict[str, str | int]:
        """Encode only scalar Chroma metadata without source text or absolute paths."""

        payload = self.model_dump(mode="python")
        section_path = payload.pop("section_path")
        payload["section_path"] = json.dumps(
            list(section_path), ensure_ascii=False, separators=(",", ":")
        )
        return {
            key: value
            for key, value in payload.items()
            if isinstance(value, (str, int))
        }

    @classmethod
    def from_chroma_metadata(cls, metadata: object) -> FlatBaselineChunkMetadata:
        """Decode the exact scalar representation without alias or repair logic."""

        if not isinstance(metadata, dict):
            raise FlatBaselineError("flat Chroma metadata must be a mapping")
        payload = dict(metadata)
        encoded_path = payload.get("section_path")
        if not isinstance(encoded_path, str):
            raise FlatBaselineError("flat Chroma section_path must be canonical JSON")
        try:
            decoded_path = json.loads(encoded_path)
        except json.JSONDecodeError as exc:
            raise FlatBaselineError("flat Chroma section_path is invalid JSON") from exc
        if not isinstance(decoded_path, list) or any(
            not isinstance(item, str) for item in decoded_path
        ):
            raise FlatBaselineError("flat Chroma section_path must be a string list")
        canonical = json.dumps(decoded_path, ensure_ascii=False, separators=(",", ":"))
        if canonical != encoded_path:
            raise FlatBaselineError("flat Chroma section_path is not canonical JSON")
        payload["section_path"] = tuple(decoded_path)
        try:
            return cls.model_validate(payload)
        except Exception as exc:
            raise FlatBaselineError(
                "flat Chroma metadata violates its contract"
            ) from exc


class FlatBaselineDocument(_StrictFrozenModel):
    """One immutable flat chunk ready for Chroma persistence."""

    schema_version: Literal["flat_baseline_document_v1"]
    content: str = Field(min_length=1)
    metadata: FlatBaselineChunkMetadata

    @model_validator(mode="after")
    def _validate_content(self) -> FlatBaselineDocument:
        if len(self.content) != self.metadata.chunk_chars:
            raise ValueError("flat chunk content length differs from metadata")
        if _sha1_text(self.content) != self.metadata.content_sha1:
            raise ValueError("flat chunk content SHA1 differs from metadata")
        return self


class FlatBaselineManifest(_StrictFrozenModel):
    """Verified identity of a non-active flat artifact used only by benchmark."""

    schema_version: Literal["flat_baseline_manifest_v2"]
    collection_name: str = Field(min_length=3)
    embedding: EmbeddingManifestIdentity
    bm25_tokenizer_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    flat_policy_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    source_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    chunk_count: int = Field(gt=0)
    chunk_id_set_sha256: str = Field(pattern=_SHA256_PATTERN)


class DocumentEmbeddingProvider(Protocol):
    """Explicit document embedding boundary used only to build a flat artifact."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class QueryEmbeddingProvider(Protocol):
    """Explicit query embedding boundary used only by the benchmark reader."""

    def embed_query(self, text: str) -> list[float]: ...


class FlatReranker(Protocol):
    """Strict reranker boundary shared with the candidate runtime."""

    def rerank(
        self, *, query: str, candidates: tuple[RerankCandidate, ...]
    ) -> Sequence[RerankScore]: ...


@dataclass(frozen=True, slots=True)
class FlatBaselineHit:
    """One reranked flat chunk and its exact provenance."""

    document: FlatBaselineDocument
    rank: int
    rerank_score: float


@dataclass(frozen=True, slots=True)
class FlatBaselineRetrievalResult:
    """Fail-fast flat retrieval outcome; empty is valid data."""

    hits: tuple[FlatBaselineHit, ...]
    vector_ms: float
    bm25_ms: float
    reranker_ms: float
    total_ms: float


def _milliseconds(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


def _validate_embedding_vector(value: object, *, dimension: int) -> list[float]:
    if not isinstance(value, list) or len(value) != dimension:
        raise FlatBaselineError("flat embedding vector dimension mismatch")
    vector: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise FlatBaselineError("flat embedding coordinate is not numeric")
        numeric = float(coordinate)
        if not math.isfinite(numeric):
            raise FlatBaselineError("flat embedding coordinate is not finite")
        vector.append(numeric)
    return vector


def _decode_document(
    identifier: object, document: object, metadata: object
) -> FlatBaselineDocument:
    if not isinstance(identifier, str) or not isinstance(document, str) or not document:
        raise FlatBaselineError("flat Chroma document payload is invalid")
    strict_metadata = FlatBaselineChunkMetadata.from_chroma_metadata(metadata)
    if strict_metadata.chunk_id != identifier:
        raise FlatBaselineError("flat Chroma ID and metadata chunk_id differ")
    try:
        return FlatBaselineDocument(
            schema_version="flat_baseline_document_v1",
            content=document,
            metadata=strict_metadata,
        )
    except Exception as exc:
        raise FlatBaselineError("flat Chroma document violates its contract") from exc


def _collection_metadata(*, manifest: FlatBaselineManifest) -> dict[str, str | int]:
    return {
        "schema_version": "flat_baseline_chroma_v2",
        "expected_dimension": manifest.embedding.dimension,
        "hnsw:space": manifest.embedding.distance_metric,
    }


def _iter_flat_collection_pages(
    *,
    collection: object,
    expected_count: int,
    page_size: int,
    include: tuple[Literal["documents", "metadatas"], ...],
) -> Iterator[tuple[tuple[str, ...], dict[str, object]]]:
    """Read every flat Chroma row in explicit bounded pages.

    Chroma translates an unbounded ``Collection.get`` into a SQL query whose
    variable count can exceed SQLite's limit for a production-sized corpus.
    Pagination is therefore a persistence protocol invariant, not a best-effort
    optimization.  Each page must be complete and disjoint; callers still
    verify their full ID-set digest after this iterator returns.
    """

    if expected_count <= 0:
        raise FlatBaselineError("flat Chroma expected count must be positive")
    if page_size <= 0:
        raise FlatBaselineError("flat Chroma read page size must be positive")
    get = getattr(collection, "get", None)
    if not callable(get):
        raise FlatBaselineError("flat Chroma collection does not expose get")

    seen_identifiers: set[str] = set()
    for offset in range(0, expected_count, page_size):
        limit = min(page_size, expected_count - offset)
        try:
            payload = get(limit=limit, offset=offset, include=list(include))
        except Exception as exc:
            raise FlatBaselineError("flat Chroma paged get failed") from exc
        if not isinstance(payload, dict):
            raise FlatBaselineError("flat Chroma paged get payload is invalid")
        raw_identifiers = payload.get("ids")
        if not isinstance(raw_identifiers, list) or len(raw_identifiers) != limit:
            raise FlatBaselineError("flat Chroma paged get cardinality mismatch")
        identifiers: list[str] = []
        for raw_identifier in raw_identifiers:
            if not isinstance(raw_identifier, str) or not raw_identifier:
                raise FlatBaselineError("flat Chroma paged get identifier is invalid")
            if raw_identifier in seen_identifiers:
                raise FlatBaselineError("flat Chroma paged get returned duplicate ID")
            identifiers.append(raw_identifier)
        seen_identifiers.update(identifiers)
        yield tuple(identifiers), dict(payload)

    if len(seen_identifiers) != expected_count:
        raise FlatBaselineError("flat Chroma paged get did not cover expected IDs")


def read_flat_collection_ids(
    *, collection: object, expected_count: int, page_size: int
) -> tuple[str, ...]:
    """Return the full persisted ID set without an unbounded Chroma query."""

    identifiers: list[str] = []
    for page_identifiers, _ in _iter_flat_collection_pages(
        collection=collection,
        expected_count=expected_count,
        page_size=page_size,
        include=(),
    ):
        identifiers.extend(page_identifiers)
    return tuple(identifiers)


def read_flat_collection_documents(
    *, collection: object, expected_count: int, page_size: int
) -> tuple[FlatBaselineDocument, ...]:
    """Decode every persisted flat document through the bounded-read contract."""

    documents: list[FlatBaselineDocument] = []
    for identifiers, payload in _iter_flat_collection_pages(
        collection=collection,
        expected_count=expected_count,
        page_size=page_size,
        include=("documents", "metadatas"),
    ):
        raw_documents = payload.get("documents")
        raw_metadatas = payload.get("metadatas")
        if not (
            isinstance(raw_documents, list)
            and isinstance(raw_metadatas, list)
            and len(raw_documents) == len(raw_metadatas) == len(identifiers)
        ):
            raise FlatBaselineError("flat Chroma paged document payload is invalid")
        documents.extend(
            _decode_document(identifier, document, metadata)
            for identifier, document, metadata in zip(
                identifiers, raw_documents, raw_metadatas, strict=True
            )
        )
    if len(documents) != expected_count:
        raise FlatBaselineError("flat Chroma paged document count mismatch")
    return tuple(documents)


def write_flat_baseline_collection(
    *,
    documents: Sequence[FlatBaselineDocument],
    persist_directory: Path,
    manifest: FlatBaselineManifest,
    embedding_provider: DocumentEmbeddingProvider,
    batch_size: int,
    max_in_flight_batches: int,
) -> None:
    """Create a new verified flat Chroma collection; existing paths are rejected."""

    if not documents:
        raise FlatBaselineError("flat baseline build requires at least one document")
    if batch_size <= 0:
        raise ValueError("flat baseline batch_size must be positive")
    if type(max_in_flight_batches) is not int or not 1 <= max_in_flight_batches <= 4:
        raise ValueError(
            "flat baseline max_in_flight_batches must be an integer from 1 to 4"
        )
    if persist_directory.exists() or persist_directory.is_symlink():
        raise FlatBaselineError(
            "flat baseline persist directory must not already exist"
        )
    persist_directory.parent.mkdir(parents=True, exist_ok=True)
    metadata = _collection_metadata(manifest=manifest)
    expected_ids = tuple(item.metadata.chunk_id for item in documents)
    if len(expected_ids) != len(set(expected_ids)):
        raise FlatBaselineError("flat baseline documents contain duplicate IDs")
    if len(expected_ids) != manifest.chunk_count:
        raise FlatBaselineError("flat baseline manifest chunk count mismatch")
    if digest_identifier_set(expected_ids) != manifest.chunk_id_set_sha256:
        raise FlatBaselineError("flat baseline manifest ID-set mismatch")
    try:
        with chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False),
        ) as client:
            if client.list_collections():
                raise FlatBaselineError(
                    "new flat baseline directory contains collections"
                )
            collection = client.create_collection(
                name=manifest.collection_name,
                metadata=metadata,
                embedding_function=None,
                get_or_create=False,
            )
            try:
                for embedded_batch in iter_bounded_document_embedding_batches(
                    texts=tuple(item.content for item in documents),
                    batch_size=batch_size,
                    max_in_flight_batches=max_in_flight_batches,
                    embed_documents=embedding_provider.embed_documents,
                ):
                    batch = tuple(
                        documents[
                            embedded_batch.batch_start : embedded_batch.batch_start
                            + embedded_batch.batch_size
                        ]
                    )
                    vectors = embedded_batch.result
                    if not isinstance(vectors, list) or len(vectors) != len(batch):
                        raise FlatBaselineError(
                            "flat baseline embedding response cardinality mismatch"
                        )
                    normalized = [
                        _validate_embedding_vector(
                            vector, dimension=manifest.embedding.dimension
                        )
                        for vector in vectors
                    ]
                    collection.add(
                        ids=[item.metadata.chunk_id for item in batch],
                        documents=[item.content for item in batch],
                        metadatas=[
                            item.metadata.to_chroma_metadata() for item in batch
                        ],
                        embeddings=normalized,
                    )
            except EmbeddingBatchExecutionError as exc:
                raise FlatBaselineError(
                    "flat baseline embedding provider failed"
                ) from exc
            actual_ids = read_flat_collection_ids(
                collection=collection,
                expected_count=manifest.chunk_count,
                page_size=batch_size,
            )
            if (
                len(actual_ids) != manifest.chunk_count
                or digest_identifier_set(actual_ids) != manifest.chunk_id_set_sha256
            ):
                raise FlatBaselineError("persisted flat baseline ID-set mismatch")
            if collection.metadata != metadata:
                raise FlatBaselineError(
                    "persisted flat baseline collection metadata mismatch"
                )
    except FlatBaselineError:
        raise
    except Exception as exc:
        raise FlatBaselineError("failed to write flat baseline Chroma") from exc


class FlatBaselineRuntime:
    """Explicit non-active flat Vector/BM25/reranker runtime for benchmarking."""

    def __init__(
        self,
        *,
        persist_directory: Path,
        manifest: FlatBaselineManifest,
        query_embedding_provider: QueryEmbeddingProvider,
        reranker: FlatReranker,
        tokenizer: Callable[[str], Sequence[str]],
        read_page_size: int,
    ) -> None:
        self._snapshot: ChromaRuntimeSnapshot | None = None
        self._canonical_persist_directory: Path | None = None
        self._canonical_sha256: str | None = None
        if persist_directory.is_symlink() or not persist_directory.is_dir():
            raise FlatBaselineError(
                "flat persist directory must be a regular directory"
            )
        self._manifest = manifest
        self._embedding_provider = query_embedding_provider
        self._reranker = reranker
        self._tokenizer = tokenizer
        try:
            self._client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_collection(
                name=manifest.collection_name, embedding_function=None
            )
        except Exception as exc:
            raise FlatBaselineError("unable to open flat baseline Chroma") from exc
        if self._collection.metadata != _collection_metadata(manifest=manifest):
            self.close()
            raise FlatBaselineError("flat baseline collection metadata mismatch")
        try:
            corpus = read_flat_collection_documents(
                collection=self._collection,
                expected_count=manifest.chunk_count,
                page_size=read_page_size,
            )
            if (
                digest_identifier_set(tuple(item.metadata.chunk_id for item in corpus))
                != manifest.chunk_id_set_sha256
            ):
                raise FlatBaselineError(
                    "flat baseline collection ID-set digest mismatch"
                )
            tokenized = [tuple(tokenizer(item.content)) for item in corpus]
            if any(not tokens for tokens in tokenized):
                raise FlatBaselineError(
                    "flat baseline tokenizer produced an empty corpus row"
                )
            self._corpus = corpus
            self._bm25 = BM25Okapi(tokenized)
        except Exception:
            self.close()
            raise

    @classmethod
    def from_canonical_artifact(
        cls,
        *,
        project_root: Path,
        persist_directory: Path,
        manifest: FlatBaselineManifest,
        query_embedding_provider: QueryEmbeddingProvider,
        reranker: FlatReranker,
        tokenizer: Callable[[str], Sequence[str]],
        read_page_size: int,
    ) -> FlatBaselineRuntime:
        """Open a disposable copy while keeping the canonical artifact immutable."""

        root = project_root.resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise FlatBaselineError("project_root must be a regular directory")
        if persist_directory.is_symlink():
            raise FlatBaselineError("flat persist directory must not be a symlink")
        canonical = persist_directory.resolve(strict=True)
        if not canonical.is_dir() or not canonical.is_relative_to(root):
            raise FlatBaselineError(
                "flat persist directory must be contained by project_root"
            )
        canonical_sha256 = sha256_path(canonical)
        snapshot = ChromaRuntimeSnapshot.create(
            index_root=root,
            source_directory=canonical,
            expected_source_sha256=canonical_sha256,
            owner_schema_version=CHROMA_RUNTIME_OWNER_SCHEMA_VERSION,
        )
        try:
            runtime = cls(
                persist_directory=snapshot.persist_directory,
                manifest=manifest,
                query_embedding_provider=query_embedding_provider,
                reranker=reranker,
                tokenizer=tokenizer,
                read_page_size=read_page_size,
            )
        except BaseException:
            snapshot.close()
            raise
        runtime._snapshot = snapshot
        runtime._canonical_persist_directory = canonical
        runtime._canonical_sha256 = canonical_sha256
        return runtime

    def close(self) -> None:
        client = getattr(self, "_client", None)
        try:
            if client is not None:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
                self._client = None
        finally:
            snapshot = self._snapshot
            if snapshot is not None:
                snapshot.close()
                self._snapshot = None
            canonical = self._canonical_persist_directory
            canonical_sha256 = self._canonical_sha256
            if canonical is not None and canonical_sha256 is not None:
                if sha256_path(canonical) != canonical_sha256:
                    raise FlatBaselineError(
                        "canonical flat artifact changed during runtime use"
                    )

    def __enter__(self) -> FlatBaselineRuntime:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _vector_candidates(
        self, *, query: str, subject: str, top_k: int
    ) -> tuple[FlatBaselineDocument, ...]:
        try:
            vector = self._embedding_provider.embed_query(query)
        except Exception as exc:
            raise FlatBaselineError("flat baseline query embedding failed") from exc
        normalized = _validate_embedding_vector(
            vector, dimension=self._manifest.embedding.dimension
        )
        try:
            payload = self._collection.query(
                query_embeddings=[normalized],
                n_results=top_k,
                where={"subject": subject},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise FlatBaselineError("flat baseline vector query failed") from exc
        ids_batches = payload.get("ids")
        docs_batches = payload.get("documents")
        metas_batches = payload.get("metadatas")
        distances_batches = payload.get("distances")
        if not (
            isinstance(ids_batches, list)
            and isinstance(docs_batches, list)
            and isinstance(metas_batches, list)
            and isinstance(distances_batches, list)
            and len(ids_batches)
            == len(docs_batches)
            == len(metas_batches)
            == len(distances_batches)
            == 1
        ):
            raise FlatBaselineError("flat vector result shape is invalid")
        ids, docs, metas, distances = (
            ids_batches[0],
            docs_batches[0],
            metas_batches[0],
            distances_batches[0],
        )
        if not (
            isinstance(ids, list)
            and isinstance(docs, list)
            and isinstance(metas, list)
            and isinstance(distances, list)
            and len(ids) == len(docs) == len(metas) == len(distances)
        ):
            raise FlatBaselineError("flat vector result cardinality is invalid")
        output: list[FlatBaselineDocument] = []
        for identifier, document, metadata, distance in zip(
            ids, docs, metas, distances, strict=True
        ):
            if (
                isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not math.isfinite(float(distance))
            ):
                raise FlatBaselineError("flat vector distance is invalid")
            decoded = _decode_document(identifier, document, metadata)
            if decoded.metadata.subject != subject:
                raise FlatBaselineError("flat vector subject filter was violated")
            output.append(decoded)
        return tuple(output)

    def _bm25_candidates(
        self, *, query: str, subject: str, top_k: int
    ) -> tuple[FlatBaselineDocument, ...]:
        try:
            tokens = tuple(self._tokenizer(query))
        except Exception as exc:
            raise FlatBaselineError("flat baseline query tokenization failed") from exc
        if not tokens:
            raise FlatBaselineError(
                "flat baseline query tokenization produced no tokens"
            )
        try:
            scores = self._bm25.get_scores(tokens)
        except Exception as exc:
            raise FlatBaselineError("flat baseline BM25 search failed") from exc
        ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))
        output: list[FlatBaselineDocument] = []
        for index, score in ranked:
            numeric = float(score)
            if not math.isfinite(numeric):
                raise FlatBaselineError("flat baseline BM25 score is invalid")
            if numeric <= 0.0:
                break
            document = self._corpus[index]
            if document.metadata.subject != subject:
                continue
            output.append(document)
            if len(output) == top_k:
                break
        return tuple(output)

    def retrieve(
        self,
        *,
        query: str,
        subject: str,
        vector_top_k: int,
        bm25_top_k: int,
        reranker_top_n: int,
    ) -> FlatBaselineRetrievalResult:
        """Run the legacy flat ordering with explicit fail-fast dependencies."""

        if (
            not query
            or query != query.strip()
            or not subject
            or subject != subject.strip()
        ):
            raise ValueError(
                "flat retrieval query and subject must be non-empty and stripped"
            )
        if min(vector_top_k, bm25_top_k, reranker_top_n) <= 0:
            raise ValueError("flat retrieval top-k values must be positive")
        total_started = perf_counter_ns()
        vector_started = perf_counter_ns()
        vector = self._vector_candidates(
            query=query, subject=subject, top_k=vector_top_k
        )
        vector_ms = _milliseconds(vector_started)
        bm25_started = perf_counter_ns()
        bm25 = self._bm25_candidates(query=query, subject=subject, top_k=bm25_top_k)
        bm25_ms = _milliseconds(bm25_started)
        merged: list[FlatBaselineDocument] = []
        content_hashes: set[str] = set()
        for document in (*vector, *bm25):
            digest = hashlib.md5(document.content.encode("utf-8")).hexdigest()
            if digest not in content_hashes:
                content_hashes.add(digest)
                merged.append(document)
        reranker_started = perf_counter_ns()
        if merged:
            submitted = tuple(
                RerankCandidate(
                    schema_version="rerank_candidate_v1",
                    child_id=document.metadata.chunk_id,
                    content=document.content,
                )
                for document in merged
            )
            try:
                scores = tuple(self._reranker.rerank(query=query, candidates=submitted))
            except Exception as exc:
                raise FlatBaselineError("flat baseline reranker failed") from exc
            by_id = {score.child_id: score for score in scores}
            if len(by_id) != len(scores) or set(by_id) != {
                item.child_id for item in submitted
            }:
                raise FlatBaselineError(
                    "flat baseline reranker response identity mismatch"
                )
            ordered = sorted(
                merged,
                key=lambda item: (
                    -by_id[item.metadata.chunk_id].score,
                    item.metadata.chunk_id,
                ),
            )[:reranker_top_n]
            hits = tuple(
                FlatBaselineHit(
                    document=document,
                    rank=index,
                    rerank_score=by_id[document.metadata.chunk_id].score,
                )
                for index, document in enumerate(ordered, start=1)
            )
        else:
            hits = ()
        reranker_ms = _milliseconds(reranker_started)
        return FlatBaselineRetrievalResult(
            hits=hits,
            vector_ms=vector_ms,
            bm25_ms=bm25_ms,
            reranker_ms=reranker_ms,
            total_ms=_milliseconds(total_started),
        )


def build_flat_baseline_manifest(
    *,
    collection_name: str,
    embedding: EmbeddingManifestIdentity,
    bm25_tokenizer_fingerprint: str,
    flat_policy_fingerprint: str,
    source_fingerprint: str,
    documents: Sequence[FlatBaselineDocument],
) -> FlatBaselineManifest:
    """Create a strict manifest only from already validated flat documents."""

    identifiers = tuple(item.metadata.chunk_id for item in documents)
    if not identifiers:
        raise FlatBaselineError("cannot manifest an empty flat baseline")
    if len(identifiers) != len(set(identifiers)):
        raise FlatBaselineError("flat baseline document IDs must be unique")
    return FlatBaselineManifest(
        schema_version="flat_baseline_manifest_v2",
        collection_name=collection_name,
        embedding=embedding,
        bm25_tokenizer_fingerprint=bm25_tokenizer_fingerprint,
        flat_policy_fingerprint=flat_policy_fingerprint,
        source_fingerprint=source_fingerprint,
        chunk_count=len(identifiers),
        chunk_id_set_sha256=digest_identifier_set(identifiers),
    )


def flat_manifest_bytes(manifest: FlatBaselineManifest) -> bytes:
    """Serialize a validated baseline manifest deterministically."""

    return canonical_json_bytes(manifest.model_dump(mode="json"))


def flat_manifest_sha256(manifest: FlatBaselineManifest) -> str:
    """Return the exact digest bound into benchmark artifacts."""

    return sha256_bytes(flat_manifest_bytes(manifest))


def compute_flat_retrieval_fingerprint(
    *,
    manifest: FlatBaselineManifest,
    vector_top_k: int,
    bm25_top_k: int,
    reranker_top_n: int,
) -> str:
    """Fingerprint every flat runtime choice that can affect ranked evidence."""

    if min(vector_top_k, bm25_top_k, reranker_top_n) <= 0:
        raise ValueError("flat retrieval top-k values must be positive")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "algorithm": "flat_vector_bm25_content_dedup_reranker_v1",
                "bm25_tokenizer_fingerprint": manifest.bm25_tokenizer_fingerprint,
                "embedding_fingerprint": manifest.embedding.fingerprint,
                "flat_policy_fingerprint": manifest.flat_policy_fingerprint,
                "reranker_top_n": reranker_top_n,
                "vector_top_k": vector_top_k,
                "bm25_top_k": bm25_top_k,
            }
        )
    )


__all__ = [
    "DocumentEmbeddingProvider",
    "FlatBaselineChunkMetadata",
    "FlatBaselineDocument",
    "FlatBaselineError",
    "FlatBaselineHit",
    "FlatBaselineManifest",
    "FlatBaselineRetrievalResult",
    "FlatBaselineRuntime",
    "QueryEmbeddingProvider",
    "build_flat_baseline_manifest",
    "compute_flat_retrieval_fingerprint",
    "flat_manifest_bytes",
    "flat_manifest_sha256",
    "make_flat_chunk_id",
    "write_flat_baseline_collection",
]
