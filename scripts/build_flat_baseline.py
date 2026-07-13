"""Build one non-active, strict flat-baseline artifact for paired RAG benchmark.

The command never reads or modifies the active Chroma directory, registry, or a
parent-child generation.  It requires a new persist directory and explicit
strict index configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.config.rag_index_config import (  # noqa: E402
    RagIndexConfig,
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.parent_child._storage_io import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
)
from src.rag.parent_child.bm25_artifact import compute_tokenizer_fingerprint  # noqa: E402
from src.rag.parent_child.builder import compute_embedding_fingerprint  # noqa: E402
from src.rag.parent_child.config_adapter import resolve_subject_chunk_policy  # noqa: E402
from src.rag.parent_child.flat_baseline import (  # noqa: E402
    FlatBaselineChunkMetadata,
    FlatBaselineDocument,
    FlatBaselineManifest,
    build_flat_baseline_manifest,
    flat_manifest_bytes,
    make_flat_chunk_id,
    write_flat_baseline_collection,
)
from src.rag.parent_child.loader import load_cleaned_source  # noqa: E402
from src.rag.parent_child.manifests import EmbeddingManifestIdentity  # noqa: E402
from src.rag.parent_child.models import SourceEntry  # noqa: E402
from src.rag.parent_child.project_paths import (  # noqa: E402
    atomic_write_project_bytes,
    require_project_file,
    resolve_project_path,
    resolve_project_root,
)
from src.rag.parent_child.provider_clients import StrictEmbeddingClient  # noqa: E402
from src.rag.parent_child.splitter import build_parent_child_bundle  # noqa: E402
from src.rag.parent_child.tokenizer import ConfiguredJiebaTokenizer  # noqa: E402
from src.rag.subject_catalog import SourceCatalogEntry, SubjectCatalog  # noqa: E402


class FlatBaselineBuildError(RuntimeError):
    """The flat artifact cannot be built without violating its strict contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline", choices=("flat-baseline",), required=True)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--persist-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--flat-build-id", required=True)
    return parser


def _doc_type(source: SourceCatalogEntry) -> str:
    """Use the exact source-type labels emitted by the candidate builder."""

    mapping = {".pdf": "pdf", ".md": "markdown", ".txt": "text"}
    try:
        return mapping[source.extension]
    except KeyError as exc:
        raise FlatBaselineBuildError(
            "catalog returned unsupported source extension"
        ) from exc


def _embedding_identity(config: RagIndexConfig) -> EmbeddingManifestIdentity:
    embedding = config.embedding
    return EmbeddingManifestIdentity(
        provider=embedding.provider,
        model=embedding.model,
        base_url_identity=embedding.base_url.rstrip("/") + embedding.endpoint_path,
        input_types=(embedding.document_input_type, embedding.query_input_type),
        fingerprint=compute_embedding_fingerprint(config),
        dimension=embedding.expected_dimension,
        distance_metric=embedding.distance_metric,
    )


def _build_documents(
    *, config: RagIndexConfig, flat_build_id: str
) -> tuple[FlatBaselineDocument, ...]:
    try:
        snapshot = SubjectCatalog(
            config=config.catalog,
            subject_policy_map=config.subject_policy_map,
        ).discover()
    except Exception as exc:
        raise FlatBaselineBuildError("SubjectCatalog discovery failed") from exc
    documents: list[FlatBaselineDocument] = []
    for subject in snapshot.subjects:
        resolved_policy = resolve_subject_chunk_policy(config, subject.subject_id)
        for source in subject.sources:
            try:
                cleaned = load_cleaned_source(
                    SourceEntry(
                        schema_version="source_entry_v1",
                        source_path=source.source_path,
                        data_root=snapshot.data_root,
                        subject=source.subject_id,
                        doc_type=_doc_type(source),
                    ),
                    resolved_policy.loader_config,
                )
                bundle = build_parent_child_bundle(
                    cleaned,
                    resolved_policy.parent_child_policy,
                    generation_id=flat_build_id,
                )
            except Exception as exc:
                raise FlatBaselineBuildError(
                    f"page-aware flat chunk build failed for source={source.source_relpath}"
                ) from exc
            for chunk_index, child in enumerate(bundle.children):
                metadata = child.metadata
                flat_metadata = FlatBaselineChunkMetadata(
                    schema_version="flat_baseline_chunk_metadata_v1",
                    chunk_id=make_flat_chunk_id(
                        doc_id=metadata.doc_id,
                        policy_id=metadata.policy_id,
                        start_char=metadata.start_char,
                        end_char=metadata.end_char,
                        content_sha1=metadata.content_sha1,
                    ),
                    doc_id=metadata.doc_id,
                    subject=metadata.subject,
                    policy_id=metadata.policy_id,
                    chunk_index=chunk_index,
                    start_char=metadata.start_char,
                    end_char=metadata.end_char,
                    chunk_chars=metadata.child_chars,
                    content_sha1=metadata.content_sha1,
                    source_file=metadata.source_file,
                    source_relpath=metadata.source_relpath,
                    source_file_sha1=metadata.source_file_sha1,
                    doc_type=metadata.doc_type,
                    section_path=metadata.section_path,
                    pagination_kind=metadata.pagination_kind,
                    page_start=metadata.page_start,
                    page_end=metadata.page_end,
                )
                documents.append(
                    FlatBaselineDocument(
                        schema_version="flat_baseline_document_v1",
                        content=child.content,
                        metadata=flat_metadata,
                    )
                )
    if not documents:
        raise FlatBaselineBuildError("flat baseline source catalog yielded no chunks")
    ids = tuple(item.metadata.chunk_id for item in documents)
    if len(ids) != len(set(ids)):
        raise FlatBaselineBuildError("flat baseline IDs are not unique")
    return tuple(documents)


def _source_fingerprint(documents: tuple[FlatBaselineDocument, ...]) -> str:
    sources = sorted(
        {
            (
                document.metadata.doc_id,
                document.metadata.source_relpath,
                document.metadata.source_file_sha1,
            )
            for document in documents
        }
    )
    return sha256_bytes(canonical_json_bytes(sources))


def _policy_fingerprint(documents: tuple[FlatBaselineDocument, ...]) -> str:
    policies = sorted(
        {
            (document.metadata.subject, document.metadata.policy_id)
            for document in documents
        }
    )
    return sha256_bytes(canonical_json_bytes(policies))


def build_flat_baseline(
    *,
    project_root: Path,
    index_config_path: Path,
    persist_directory: Path,
    manifest_output: Path,
    collection_name: str,
    flat_build_id: str,
) -> FlatBaselineManifest:
    """Build one isolated flat artifact and its strict manifest without activation."""

    root = resolve_project_root(project_root)
    config_path = require_project_file(root, index_config_path)
    persist = resolve_project_path(root, persist_directory, must_exist=False)
    manifest_path = resolve_project_path(root, manifest_output, must_exist=False)
    if persist.exists():
        raise FlatBaselineBuildError("flat baseline persist-dir must not already exist")
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if not collection_name or collection_name != collection_name.strip():
        raise ValueError("collection-name must be non-empty and stripped")
    if not flat_build_id or flat_build_id != flat_build_id.strip():
        raise ValueError("flat-build-id must be non-empty and stripped")
    config = resolve_rag_index_config_paths(
        load_rag_index_config(config_path), project_root=root
    )
    if config.catalog.symlink_policy != "reject":
        raise FlatBaselineBuildError(
            "flat baseline tooling requires catalog symlink_policy=reject"
        )
    if persist == config.storage.index_root or persist.is_relative_to(
        config.storage.index_root
    ):
        raise FlatBaselineBuildError(
            "flat baseline persist-dir must be outside index_root"
        )
    documents = _build_documents(config=config, flat_build_id=flat_build_id)
    tokenizer = ConfiguredJiebaTokenizer(config=config.bm25)
    manifest = build_flat_baseline_manifest(
        collection_name=collection_name,
        embedding=_embedding_identity(config),
        bm25_tokenizer_fingerprint=compute_tokenizer_fingerprint(
            tokenizer_name=config.bm25.tokenizer,
            tokenizer_version=config.bm25.tokenizer_version,
            dictionary_sha256=config.bm25.dictionary_hash,
        ),
        flat_policy_fingerprint=_policy_fingerprint(documents),
        source_fingerprint=_source_fingerprint(documents),
        documents=documents,
    )
    embedding = StrictEmbeddingClient.production(config=config.embedding)
    try:
        write_flat_baseline_collection(
            documents=documents,
            persist_directory=persist,
            manifest=manifest,
            embedding_provider=embedding,
            batch_size=config.embedding.batch_size,
            max_in_flight_batches=config.embedding.max_in_flight_batches,
        )
    finally:
        embedding.close()
        # Instantiate the checked tokenizer before writing the manifest.  This
        # makes its runtime dictionary identity a build precondition even though
        # the flat BM25 index itself is rebuilt in memory at benchmark time.
        del tokenizer
    atomic_write_project_bytes(
        root,
        manifest_path,
        flat_manifest_bytes(manifest),
        overwrite=False,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_flat_baseline(
        project_root=args.project_root,
        index_config_path=args.index_config,
        persist_directory=args.persist_dir,
        manifest_output=args.manifest_output,
        collection_name=args.collection_name,
        flat_build_id=args.flat_build_id,
    )
    print(
        "Flat baseline built: "
        f"collection={manifest.collection_name}, chunks={manifest.chunk_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
