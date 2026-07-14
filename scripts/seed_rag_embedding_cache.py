"""Seed a strict resumable embedding cache from one verified Flat artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.config.rag_index_config import (  # noqa: E402
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.parent_child._storage_io import sha256_path  # noqa: E402
from src.rag.parent_child.bm25_artifact import digest_identifier_set  # noqa: E402
from src.rag.parent_child.builder import compute_embedding_fingerprint  # noqa: E402
from src.rag.parent_child.chroma_runtime_snapshot import (  # noqa: E402
    CHROMA_RUNTIME_OWNER_SCHEMA_VERSION,
    ChromaRuntimeSnapshot,
)
from src.rag.parent_child.embedding_cache import (  # noqa: E402
    EMBEDDING_CACHE_SCHEMA_VERSION,
    SqliteEmbeddingCache,
)
from src.rag.parent_child.flat_baseline import (  # noqa: E402
    FlatBaselineChunkMetadata,
    FlatBaselineDocument,
    FlatBaselineManifest,
)
from src.rag.parent_child.project_paths import (  # noqa: E402
    atomic_write_project_bytes,
    require_project_directory,
    require_project_file,
    resolve_project_path,
    resolve_project_root,
)


class EmbeddingCacheSeedError(RuntimeError):
    """The Flat artifact cannot be safely projected into the exact cache."""


class EmbeddingCacheSeedReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: str = Field(min_length=1)
    flat_chunk_count: int = Field(gt=0)
    unique_cached_content_count: int = Field(gt=0)
    duplicate_content_count: int = Field(ge=0)
    ambiguous_content_count: int = Field(ge=0)
    ambiguous_flat_row_count: int = Field(ge=0)
    embedding_fingerprint: str = Field(min_length=64, max_length=64)
    dimension: int = Field(gt=0)
    cache_path: str = Field(min_length=1)
    origin_counts: dict[str, int]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--flat-persist-dir", type=Path, required=True)
    parser.add_argument("--flat-manifest", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--read-page-size", type=int, required=True)
    parser.add_argument("--busy-timeout-seconds", type=float, required=True)
    return parser


def _vector_rows(value: object, *, expected_count: int) -> tuple[object, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or len(value) != expected_count:
        raise EmbeddingCacheSeedError("Flat embedding page has invalid cardinality")
    return tuple(value)


def _validate_vector(value: object, *, dimension: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EmbeddingCacheSeedError("Flat embedding row must be a sequence")
    if len(value) != dimension:
        raise EmbeddingCacheSeedError("Flat embedding row has the wrong dimension")
    vector: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise EmbeddingCacheSeedError("Flat embedding coordinate is not numeric")
        normalized = float(coordinate)
        if not math.isfinite(normalized):
            raise EmbeddingCacheSeedError("Flat embedding coordinate is not finite")
        vector.append(normalized)
    return vector


def seed_cache(
    *,
    project_root: Path,
    index_config_path: Path,
    flat_persist_directory: Path,
    flat_manifest_path: Path,
    cache_path: Path,
    output_path: Path,
    read_page_size: int,
    busy_timeout_seconds: float,
) -> EmbeddingCacheSeedReport:
    root = resolve_project_root(project_root)
    config_path = require_project_file(root, index_config_path)
    persist = require_project_directory(root, flat_persist_directory)
    manifest_path = require_project_file(root, flat_manifest_path)
    cache_output = resolve_project_path(root, cache_path, must_exist=False)
    report_output = resolve_project_path(root, output_path, must_exist=False)
    if cache_output.exists() or report_output.exists():
        raise FileExistsError("cache and report outputs must be new")
    if read_page_size <= 0 or busy_timeout_seconds <= 0:
        raise ValueError("page size and busy timeout must be positive")
    config = resolve_rag_index_config_paths(
        load_rag_index_config(config_path),
        project_root=root,
    )
    manifest = FlatBaselineManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    expected_fingerprint = compute_embedding_fingerprint(config)
    if (
        manifest.embedding.fingerprint != expected_fingerprint
        or manifest.embedding.dimension != config.embedding.expected_dimension
        or manifest.embedding.distance_metric != config.embedding.distance_metric
    ):
        raise EmbeddingCacheSeedError(
            "Flat embedding identity differs from the strict index config"
        )

    source_digest = sha256_path(persist)
    with (
        ChromaRuntimeSnapshot.create(
            index_root=root,
            source_directory=persist,
            expected_source_sha256=source_digest,
            owner_schema_version=CHROMA_RUNTIME_OWNER_SCHEMA_VERSION,
        ) as snapshot,
        SqliteEmbeddingCache.create(
            project_root=root,
            cache_path=cache_output,
            schema_version=EMBEDDING_CACHE_SCHEMA_VERSION,
            embedding_fingerprint=expected_fingerprint,
            dimension=config.embedding.expected_dimension,
            busy_timeout_seconds=busy_timeout_seconds,
        ) as cache,
    ):
        client = chromadb.PersistentClient(
            path=str(snapshot.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            collection = client.get_collection(
                manifest.collection_name,
                embedding_function=None,
            )
            if collection.count() != manifest.chunk_count:
                raise EmbeddingCacheSeedError("Flat collection count mismatch")
            identifiers: list[str] = []
            for offset in range(0, manifest.chunk_count, read_page_size):
                limit = min(read_page_size, manifest.chunk_count - offset)
                payload = collection.get(
                    limit=limit,
                    offset=offset,
                    include=["documents", "metadatas", "embeddings"],
                )
                ids = payload.get("ids")
                documents = payload.get("documents")
                metadatas = payload.get("metadatas")
                if not (
                    isinstance(ids, list)
                    and isinstance(documents, list)
                    and isinstance(metadatas, list)
                    and len(ids) == len(documents) == len(metadatas) == limit
                ):
                    raise EmbeddingCacheSeedError("Flat page shape is invalid")
                vectors = _vector_rows(
                    payload.get("embeddings"),
                    expected_count=limit,
                )
                strict_documents: list[str] = []
                strict_vectors: list[list[float]] = []
                for identifier, document, metadata, vector in zip(
                    ids,
                    documents,
                    metadatas,
                    vectors,
                    strict=True,
                ):
                    if not isinstance(identifier, str) or not isinstance(document, str):
                        raise EmbeddingCacheSeedError("Flat page identity is invalid")
                    strict_metadata = FlatBaselineChunkMetadata.from_chroma_metadata(
                        metadata
                    )
                    strict_document = FlatBaselineDocument(
                        schema_version="flat_baseline_document_v1",
                        content=document,
                        metadata=strict_metadata,
                    )
                    if strict_metadata.chunk_id != identifier:
                        raise EmbeddingCacheSeedError("Flat ID/metadata mismatch")
                    identifiers.append(identifier)
                    strict_documents.append(strict_document.content)
                    strict_vectors.append(
                        _validate_vector(
                            vector,
                            dimension=config.embedding.expected_dimension,
                        )
                    )
                cache.seed_flat_many(
                    texts=strict_documents,
                    vectors=strict_vectors,
                )
        finally:
            client.close()
        if len(identifiers) != len(set(identifiers)):
            raise EmbeddingCacheSeedError("Flat collection contains duplicate IDs")
        if digest_identifier_set(identifiers) != manifest.chunk_id_set_sha256:
            raise EmbeddingCacheSeedError("Flat collection ID digest mismatch")
        origin_counts = cache.counts_by_origin()
        unique_count = sum(origin_counts.values())
        ambiguous_count, ambiguous_rows = cache.ambiguous_counts()
        cache.verify_integrity()

    report = EmbeddingCacheSeedReport(
        schema_version="embedding_cache_seed_report_v1",
        flat_chunk_count=manifest.chunk_count,
        unique_cached_content_count=unique_count,
        duplicate_content_count=manifest.chunk_count - unique_count,
        ambiguous_content_count=ambiguous_count,
        ambiguous_flat_row_count=ambiguous_rows,
        embedding_fingerprint=expected_fingerprint,
        dimension=config.embedding.expected_dimension,
        cache_path=cache_output.relative_to(root).as_posix(),
        origin_counts=origin_counts,
    )
    atomic_write_project_bytes(
        root,
        report_output,
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
        overwrite=False,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = seed_cache(
        project_root=args.project_root,
        index_config_path=args.index_config,
        flat_persist_directory=args.flat_persist_dir,
        flat_manifest_path=args.flat_manifest,
        cache_path=args.cache_path,
        output_path=args.output,
        read_page_size=args.read_page_size,
        busy_timeout_seconds=args.busy_timeout_seconds,
    )
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
