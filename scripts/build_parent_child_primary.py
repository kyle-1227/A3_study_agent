"""Build a Parent--Child primary directly into staging and atomically publish it."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.config.rag_index_config import (  # noqa: E402
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.parent_child._storage_io import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
    validate_generation_id,
)
from src.rag.parent_child.bm25_artifact import (  # noqa: E402
    Bm25CorpusRow,
    write_subject_bm25_artifact,
)
from src.rag.parent_child.chroma_children import write_child_chroma_artifact  # noqa: E402
from src.rag.parent_child.config_adapter import (  # noqa: E402
    resolve_subject_chunk_policy,
    validate_configured_ocr_runtimes,
    validate_configured_ocr_source_inventory,
)
from src.rag.parent_child.loader import load_cleaned_source  # noqa: E402
from src.rag.parent_child.manifests import (  # noqa: E402
    PolicyManifest,
    PolicyManifestSet,
    SubjectManifest,
    SubjectManifestEntry,
    write_strict_model,
)
from src.rag.parent_child.models import ChildDocument, ParentRecord, SourceEntry  # noqa: E402
from src.rag.parent_child.parent_store import create_parent_store  # noqa: E402
from src.rag.parent_child.primary_runtime import (  # noqa: E402
    PrimaryIndexWorkspace,
    primary_metadata_from_config,
    validate_primary_revision,
)
from src.rag.parent_child.provider_clients import StrictEmbeddingClient  # noqa: E402
from src.rag.parent_child.splitter import build_parent_child_bundle  # noqa: E402
from src.rag.parent_child.tokenizer import ConfiguredJiebaTokenizer  # noqa: E402
from src.rag.subject_catalog import SubjectCatalog  # noqa: E402


class PrimaryBuildError(RuntimeError):
    """A direct primary build failed before atomic publication."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--artifact-identity", required=True)
    return parser


def _contained_file(root: Path, value: Path) -> Path:
    path = value if value.is_absolute() else root / value
    if path.is_symlink():
        raise ValueError("index config must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError("index config must be a project-contained file")
    return resolved


def _tokens(tokenizer: ConfiguredJiebaTokenizer, text: str) -> tuple[str, ...]:
    raw = tokenizer(text)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise PrimaryBuildError("BM25 tokenizer returned an invalid sequence")
    result = tuple(raw)
    if not result or any(
        not isinstance(item, str) or not item.strip() for item in result
    ):
        raise PrimaryBuildError("every child requires non-empty BM25 tokens")
    return result


def _source_digest(items: Sequence[tuple[str, str]]) -> str:
    return sha256_bytes(canonical_json_bytes(sorted(items)))


def build_primary(
    *,
    project_root: Path,
    index_config_path: Path,
    build_id: str,
    artifact_identity: str,
) -> int:
    root = project_root.resolve(strict=True)
    config = resolve_rag_index_config_paths(
        load_rag_index_config(_contained_file(root, index_config_path)),
        project_root=root,
    )
    identity = validate_generation_id(artifact_identity)
    validate_configured_ocr_runtimes(config)
    workspace = PrimaryIndexWorkspace.create(
        index_root=config.storage.index_root,
        build_id=build_id,
    )
    embedding = StrictEmbeddingClient.production(config=config.embedding)
    try:
        tokenizer = ConfiguredJiebaTokenizer(config=config.bm25)
        catalog = SubjectCatalog(
            config=config.catalog,
            subject_policy_map=config.subject_policy_map,
        ).discover()
        validate_configured_ocr_source_inventory(
            config,
            tuple(
                source.source_relpath
                for subject in catalog.subjects
                for source in subject.sources
            ),
        )
        policies: dict[str, PolicyManifest] = {}
        parents: list[ParentRecord] = []
        children: list[ChildDocument] = []
        source_hashes: dict[str, list[tuple[str, str]]] = {
            subject.subject_id: [] for subject in catalog.subjects
        }
        parent_counts = {subject.subject_id: 0 for subject in catalog.subjects}
        child_counts = {subject.subject_id: 0 for subject in catalog.subjects}
        for subject in catalog.subjects:
            resolved_policy = resolve_subject_chunk_policy(config, subject.subject_id)
            policies[subject.policy_id] = resolved_policy.policy_manifest
            for source in subject.sources:
                cleaned = load_cleaned_source(
                    SourceEntry(
                        schema_version="source_entry_v1",
                        source_path=source.source_path,
                        data_root=catalog.data_root,
                        subject=subject.subject_id,
                        doc_type={
                            ".pdf": "pdf",
                            ".md": "markdown",
                            ".txt": "text",
                        }[source.extension],
                    ),
                    resolved_policy.loader_config,
                )
                bundle = build_parent_child_bundle(
                    cleaned,
                    resolved_policy.parent_child_policy,
                    identity,
                )
                parents.extend(bundle.parents)
                children.extend(bundle.children)
                source_hashes[subject.subject_id].append(
                    (cleaned.source_relpath, cleaned.source_file_sha1)
                )
        if not parents or not children:
            raise PrimaryBuildError("primary build contains no parents or children")
        create_parent_store(
            workspace.staging_path,
            "parents.sqlite",
            parents,
            store_schema_version=config.storage.parent_store_schema_version,
            expected_generation_id=identity,
            busy_timeout_seconds=config.storage.parent_store_busy_timeout_seconds,
        )
        children_by_subject: dict[str, list[ChildDocument]] = {
            subject.subject_id: [] for subject in catalog.subjects
        }
        for child in children:
            children_by_subject[child.metadata.subject].append(child)
            child_counts[child.metadata.subject] += 1
        for parent in parents:
            parent_counts[parent.subject] += 1
        for subject_id, subject_children in sorted(children_by_subject.items()):
            rows = tuple(
                Bm25CorpusRow(
                    schema_version="bm25_row_v1",
                    generation_id=identity,
                    subject=subject_id,
                    child_id=child.metadata.child_id,
                    tokens=_tokens(tokenizer, child.content),
                )
                for child in sorted(
                    subject_children,
                    key=lambda item: item.metadata.child_id,
                )
            )
            write_subject_bm25_artifact(
                workspace.staging_path,
                f"bm25/{subject_id}.jsonl",
                f"bm25/{subject_id}.manifest.json",
                rows,
                manifest_schema_version="bm25_manifest_v1",
                expected_generation_id=identity,
                expected_subject=subject_id,
                tokenizer_name=config.bm25.tokenizer,
                tokenizer_version=config.bm25.tokenizer_version,
                dictionary_sha256=config.bm25.dictionary_hash,
            )
        write_child_chroma_artifact(
            children,
            generation_staging_root=workspace.staging_path,
            persist_directory=workspace.staging_path / "chroma_children",
            generation_id=identity,
            collection_name=config.storage.collection_name,
            distance_metric=config.embedding.distance_metric,
            expected_dimension=config.embedding.expected_dimension,
            batch_size=config.embedding.batch_size,
            max_in_flight_batches=config.embedding.max_in_flight_batches,
            embedding_provider=embedding,
        )
        policy_set = PolicyManifestSet(
            schema_version="policy_manifest_set_v1",
            policies=tuple(policies[key] for key in sorted(policies)),
        )
        write_strict_model(
            workspace.staging_path,
            "policy_manifest.json",
            policy_set,
            overwrite=False,
        )
        subject_manifest = SubjectManifest(
            schema_version="subject_manifest_v1",
            generation_id=identity,
            entries=tuple(
                SubjectManifestEntry(
                    subject_id=subject.subject_id,
                    directory_relpath=subject.directory_path.relative_to(
                        catalog.data_root
                    ).as_posix(),
                    source_file_count=len(subject.sources),
                    source_manifest_sha256=_source_digest(
                        source_hashes[subject.subject_id]
                    ),
                    policy_id=subject.policy_id,
                    parent_count=parent_counts[subject.subject_id],
                    child_count=child_counts[subject.subject_id],
                    exclusion_state="active",
                    exclusion_reason="",
                )
                for subject in catalog.subjects
            ),
        )
        write_strict_model(
            workspace.staging_path,
            "subject_manifest.json",
            subject_manifest,
            overwrite=False,
        )
        metadata = primary_metadata_from_config(
            config,
            primary_revision=workspace.next_revision(),
            artifact_identity=identity,
            available_subjects=tuple(
                subject.subject_id for subject in catalog.subjects
            ),
            built_at_utc=datetime.now(UTC),
        )
        result = workspace.publish(
            metadata=metadata,
            validate_staging=lambda artifact_root, primary_metadata: (
                validate_primary_revision(
                    config=config,
                    artifact_root=artifact_root,
                    metadata=primary_metadata,
                )
            ),
        )
        return result.primary_revision
    finally:
        embedding.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    revision = build_primary(
        project_root=args.project_root,
        index_config_path=args.index_config,
        build_id=args.build_id,
        artifact_identity=args.artifact_identity,
    )
    print(f"Parent--Child primary published: revision={revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
