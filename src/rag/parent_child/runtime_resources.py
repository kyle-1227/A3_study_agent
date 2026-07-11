"""Generation-bound Chroma and subject-local BM25 runtime resources."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from threading import RLock
from typing import Protocol

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from src.rag.parent_child.bm25_artifact import (
    Bm25CorpusRow,
    read_subject_bm25_artifact,
)
from src.rag.parent_child.models import ChildDocument, ChildMetadata
from src.rag.parent_child.retrieval import (
    ChildReranker,
    ChildSearchCandidate,
    ParentHydrator,
    RetrievalChannelError,
    RetrievalInvariantError,
    RetrievalProtocolError,
)


class RuntimeResourceError(RuntimeError):
    """A sealed generation resource cannot be loaded or validated."""


class QueryEmbeddingProvider(Protocol):
    """Provider-neutral query embedding boundary."""

    def embed_query(self, text: str) -> list[float]: ...


class ChildDocumentLookup(Protocol):
    """Exact child hydration boundary shared by subject-local BM25 indexes."""

    def get_children(self, child_ids: Sequence[str]) -> tuple[ChildDocument, ...]: ...


def _validate_query_vector(value: object, *, expected_dimension: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_dimension:
        raise RetrievalProtocolError("query embedding dimension mismatch")
    result: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise RetrievalProtocolError("query embedding coordinates must be numeric")
        normalized = float(coordinate)
        if not math.isfinite(normalized):
            raise RetrievalProtocolError("query embedding coordinates must be finite")
        result.append(normalized)
    return result


def _decode_child(document: object, metadata: object) -> ChildDocument:
    if not isinstance(document, str) or not document:
        raise RetrievalProtocolError("Chroma child document must be non-empty text")
    if not isinstance(metadata, dict):
        raise RetrievalProtocolError("Chroma child metadata must be a mapping")
    try:
        strict_metadata = ChildMetadata.from_chroma_metadata(metadata)
        return ChildDocument(
            schema_version="child_document_v1",
            content=document,
            metadata=strict_metadata,
        )
    except (TypeError, ValueError) as exc:
        raise RetrievalProtocolError(
            "persisted Chroma child contract is invalid"
        ) from exc


class ChromaChildSearchChannel:
    """Exact-generation vector search and child lookup over a sealed collection."""

    def __init__(
        self,
        *,
        persist_directory: Path,
        collection_name: str,
        generation_id: str,
        expected_dimension: int,
        distance_metric: str,
        query_embedding_provider: QueryEmbeddingProvider,
    ) -> None:
        if not persist_directory.is_absolute():
            raise RuntimeResourceError("Chroma persist_directory must be absolute")
        if persist_directory.is_symlink() or not persist_directory.is_dir():
            raise RuntimeResourceError(
                "Chroma persist_directory must be a non-symlink directory"
            )
        if not collection_name or not generation_id:
            raise RuntimeResourceError("collection_name and generation_id are required")
        if expected_dimension <= 0:
            raise RuntimeResourceError("expected_dimension must be positive")
        if distance_metric not in {"cosine", "l2", "ip"}:
            raise RuntimeResourceError("unsupported Chroma distance metric")
        self._generation_id = generation_id
        self._expected_dimension = expected_dimension
        self._embedding_provider = query_embedding_provider
        try:
            self._client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_collection(
                collection_name,
                embedding_function=None,
            )
        except Exception as exc:
            raise RuntimeResourceError("unable to open sealed child Chroma") from exc
        expected_metadata = {
            "schema_version": "chroma_children_v1",
            "generation_id": generation_id,
            "expected_dimension": expected_dimension,
            "hnsw:space": distance_metric,
        }
        if self._collection.metadata != expected_metadata:
            self.close()
            raise RuntimeResourceError("child Chroma collection metadata mismatch")

    def close(self) -> None:
        """Release the Chroma client context when supported by this version."""

        exit_method = getattr(self._client, "__exit__", None)
        if callable(exit_method):
            exit_method(None, None, None)

    def _embed_query(self, query: str) -> list[float]:
        try:
            raw = self._embedding_provider.embed_query(query)
        except Exception as exc:
            raise RetrievalChannelError("query embedding provider failed") from exc
        return _validate_query_vector(raw, expected_dimension=self._expected_dimension)

    def search(
        self,
        *,
        query: str,
        subject: str,
        generation_id: str,
        top_k: int,
    ) -> tuple[ChildSearchCandidate, ...]:
        if generation_id != self._generation_id:
            raise RetrievalInvariantError("vector generation does not match resource")
        if not query or not subject or top_k <= 0:
            raise RetrievalProtocolError(
                "vector query, subject, and top_k are required"
            )
        vector = self._embed_query(query)
        try:
            raw = self._collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where={
                    "$and": [
                        {"subject": {"$eq": subject}},
                        {"generation_id": {"$eq": generation_id}},
                    ]
                },
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RetrievalChannelError("child vector search failed") from exc
        ids = raw.get("ids")
        documents = raw.get("documents")
        metadatas = raw.get("metadatas")
        distances = raw.get("distances")
        if not all(
            isinstance(value, list) and len(value) == 1
            for value in (ids, documents, metadatas, distances)
        ):
            raise RetrievalProtocolError("Chroma query returned invalid batch shape")
        row_ids, row_documents, row_metadatas, row_distances = (
            ids[0],
            documents[0],
            metadatas[0],
            distances[0],
        )
        if not (
            isinstance(row_ids, list)
            and isinstance(row_documents, list)
            and isinstance(row_metadatas, list)
            and isinstance(row_distances, list)
            and len(row_ids)
            == len(row_documents)
            == len(row_metadatas)
            == len(row_distances)
        ):
            raise RetrievalProtocolError("Chroma query result cardinality mismatch")
        seen: set[str] = set()
        candidates: list[ChildSearchCandidate] = []
        for child_id, document, metadata, distance in zip(
            row_ids,
            row_documents,
            row_metadatas,
            row_distances,
            strict=True,
        ):
            if not isinstance(child_id, str) or child_id in seen:
                raise RetrievalProtocolError("Chroma query child IDs are invalid")
            if isinstance(distance, bool) or not isinstance(distance, (int, float)):
                raise RetrievalProtocolError("Chroma distance must be numeric")
            normalized_distance = float(distance)
            if not math.isfinite(normalized_distance):
                raise RetrievalProtocolError("Chroma distance must be finite")
            child = _decode_child(document, metadata)
            if child.metadata.child_id != child_id:
                raise RetrievalInvariantError("Chroma ID and metadata child_id differ")
            if child.metadata.subject != subject:
                raise RetrievalInvariantError(
                    "Chroma exact-subject filter was violated"
                )
            if child.metadata.generation_id != generation_id:
                raise RetrievalInvariantError("Chroma generation filter was violated")
            seen.add(child_id)
            candidates.append(
                ChildSearchCandidate(
                    schema_version="child_search_candidate_v1",
                    document=child,
                    raw_score=-normalized_distance,
                )
            )
        return tuple(candidates)

    def get_children(self, child_ids: Sequence[str]) -> tuple[ChildDocument, ...]:
        requested = tuple(child_ids)
        if not requested or any(not child_id for child_id in requested):
            raise RetrievalProtocolError("child lookup requires non-empty IDs")
        if len(requested) != len(set(requested)):
            raise RetrievalProtocolError("child lookup IDs must be unique")
        try:
            raw = self._collection.get(
                ids=list(requested),
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise RetrievalChannelError("child Chroma lookup failed") from exc
        ids = raw.get("ids")
        documents = raw.get("documents")
        metadatas = raw.get("metadatas")
        if not (
            isinstance(ids, list)
            and isinstance(documents, list)
            and isinstance(metadatas, list)
            and len(ids) == len(documents) == len(metadatas)
        ):
            raise RetrievalProtocolError("child Chroma lookup shape is invalid")
        by_id: dict[str, ChildDocument] = {}
        for child_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            if not isinstance(child_id, str) or child_id in by_id:
                raise RetrievalProtocolError("child Chroma lookup IDs are invalid")
            child = _decode_child(document, metadata)
            if child.metadata.child_id != child_id:
                raise RetrievalInvariantError("child lookup ID and metadata differ")
            by_id[child_id] = child
        if set(by_id) != set(requested):
            raise RetrievalInvariantError("child Chroma lookup ID set mismatch")
        return tuple(by_id[child_id] for child_id in requested)


class SubjectBm25SearchChannel:
    """One generation- and subject-bound in-memory BM25Okapi resource."""

    def __init__(
        self,
        *,
        generation_id: str,
        subject: str,
        rows: Sequence[Bm25CorpusRow],
        tokenizer: Callable[[str], Sequence[str]],
        child_lookup: ChildDocumentLookup,
    ) -> None:
        if not rows:
            raise RuntimeResourceError("subject BM25 corpus must not be empty")
        if any(
            row.generation_id != generation_id or row.subject != subject for row in rows
        ):
            raise RuntimeResourceError("subject BM25 row identity mismatch")
        child_ids = tuple(row.child_id for row in rows)
        if len(child_ids) != len(set(child_ids)):
            raise RuntimeResourceError(
                "subject BM25 corpus contains duplicate child IDs"
            )
        self._generation_id = generation_id
        self._subject = subject
        self._rows = tuple(rows)
        self._tokenizer = tokenizer
        self._child_lookup = child_lookup
        try:
            self._index = BM25Okapi([list(row.tokens) for row in self._rows])
        except Exception as exc:
            raise RuntimeResourceError(
                "unable to construct subject BM25 index"
            ) from exc

    @classmethod
    def load(
        cls,
        *,
        generation_root: Path,
        manifest_relative_path: str,
        manifest_schema_version: str,
        generation_id: str,
        subject: str,
        tokenizer_fingerprint: str,
        tokenizer: Callable[[str], Sequence[str]],
        child_lookup: ChildDocumentLookup,
    ) -> SubjectBm25SearchChannel:
        _, rows = read_subject_bm25_artifact(
            generation_root,
            manifest_relative_path,
            expected_manifest_schema_version=manifest_schema_version,
            expected_generation_id=generation_id,
            expected_subject=subject,
            expected_tokenizer_fingerprint=tokenizer_fingerprint,
        )
        return cls(
            generation_id=generation_id,
            subject=subject,
            rows=rows,
            tokenizer=tokenizer,
            child_lookup=child_lookup,
        )

    def search(
        self,
        *,
        query: str,
        subject: str,
        generation_id: str,
        top_k: int,
    ) -> tuple[ChildSearchCandidate, ...]:
        if generation_id != self._generation_id or subject != self._subject:
            raise RetrievalInvariantError("BM25 request does not match its resource")
        if not query or top_k <= 0:
            raise RetrievalProtocolError("BM25 query and positive top_k are required")
        try:
            raw_tokens = self._tokenizer(query)
        except Exception as exc:
            raise RetrievalChannelError("BM25 query tokenization failed") from exc
        if isinstance(raw_tokens, (str, bytes)) or not isinstance(raw_tokens, Sequence):
            raise RetrievalProtocolError("BM25 tokenizer must return a token sequence")
        tokens = tuple(raw_tokens)
        if any(not isinstance(token, str) or not token.strip() for token in tokens):
            raise RetrievalProtocolError("BM25 query tokens must be non-empty strings")
        if not tokens:
            return ()
        try:
            raw_scores = self._index.get_scores(list(tokens))
        except Exception as exc:
            raise RetrievalChannelError("BM25 scoring failed") from exc
        if len(raw_scores) != len(self._rows):
            raise RetrievalProtocolError("BM25 score cardinality mismatch")
        query_token_set = set(tokens)
        ranked: list[tuple[float, str]] = []
        for row, raw_score in zip(self._rows, raw_scores, strict=True):
            score = float(raw_score)
            if not math.isfinite(score):
                raise RetrievalProtocolError("BM25 score must be finite")
            if query_token_set.isdisjoint(row.tokens):
                continue
            ranked.append((score, row.child_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = tuple(ranked[:top_k])
        if not selected:
            return ()
        documents = self._child_lookup.get_children(
            tuple(child_id for _, child_id in selected)
        )
        if tuple(document.metadata.child_id for document in documents) != tuple(
            child_id for _, child_id in selected
        ):
            raise RetrievalInvariantError("BM25 child lookup order/identity mismatch")
        return tuple(
            ChildSearchCandidate(
                schema_version="child_search_candidate_v1",
                document=document,
                raw_score=score,
            )
            for (score, _), document in zip(selected, documents, strict=True)
        )


class SubjectBm25Router:
    """Fail-fast exact-subject dispatcher; unknown subjects never search all data."""

    def __init__(self, channels: dict[str, SubjectBm25SearchChannel]) -> None:
        if not channels:
            raise RuntimeResourceError("at least one subject BM25 channel is required")
        self._channels = dict(channels)

    def search(
        self,
        *,
        query: str,
        subject: str,
        generation_id: str,
        top_k: int,
    ) -> tuple[ChildSearchCandidate, ...]:
        try:
            channel = self._channels[subject]
        except KeyError as exc:
            raise RetrievalInvariantError("unknown subject BM25 request") from exc
        return channel.search(
            query=query,
            subject=subject,
            generation_id=generation_id,
            top_k=top_k,
        )


@dataclass(frozen=True, slots=True)
class GenerationResources:
    """All resources pinned to one immutable generation/fingerprint."""

    generation_id: str
    manifest_fingerprint: str
    vector: ChromaChildSearchChannel
    bm25: SubjectBm25Router
    reranker: ChildReranker
    parents: ParentHydrator

    def close(self) -> None:
        self.vector.close()
        close_parent = getattr(self.parents, "close", None)
        if callable(close_parent):
            close_parent()


class GenerationResourceCache:
    """Thread-safe cache keyed by generation and manifest fingerprint."""

    def __init__(
        self,
        loader: Callable[[str, str], GenerationResources],
    ) -> None:
        self._loader = loader
        self._lock = RLock()
        self._resources: dict[tuple[str, str], GenerationResources] = {}

    def get(self, generation_id: str, manifest_fingerprint: str) -> GenerationResources:
        if not generation_id or not manifest_fingerprint:
            raise RuntimeResourceError(
                "generation_id and manifest_fingerprint are required"
            )
        key = (generation_id, manifest_fingerprint)
        with self._lock:
            existing = self._resources.get(key)
            if existing is not None:
                return existing
            loaded = self._loader(generation_id, manifest_fingerprint)
            if (
                loaded.generation_id != generation_id
                or loaded.manifest_fingerprint != manifest_fingerprint
            ):
                loaded.close()
                raise RuntimeResourceError(
                    "resource loader returned wrong cache identity"
                )
            self._resources[key] = loaded
            return loaded

    def close_generation(self, generation_id: str) -> None:
        with self._lock:
            keys = [key for key in self._resources if key[0] == generation_id]
            resources = [self._resources.pop(key) for key in keys]
        for resource in resources:
            resource.close()

    def close_all(self) -> None:
        with self._lock:
            resources = tuple(self._resources.values())
            self._resources.clear()
        for resource in resources:
            resource.close()


__all__ = [
    "ChromaChildSearchChannel",
    "ChildDocumentLookup",
    "GenerationResourceCache",
    "GenerationResources",
    "QueryEmbeddingProvider",
    "RuntimeResourceError",
    "SubjectBm25Router",
    "SubjectBm25SearchChannel",
]
