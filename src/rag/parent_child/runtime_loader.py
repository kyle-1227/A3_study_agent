"""Load fully verified generation resources and construct strict retrieval runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.config.rag_index_config import RagIndexConfig
from src.rag.parent_child._storage_io import canonical_json_bytes, sha256_bytes
from src.rag.parent_child.bm25_artifact import compute_tokenizer_fingerprint
from src.rag.parent_child.builder import compute_embedding_fingerprint
from src.rag.parent_child.generation import validate_sealed_generation
from src.rag.parent_child.manifests import (
    GenerationManifest,
    PolicyManifestSet,
    SubjectManifest,
    read_strict_model,
)
from src.rag.parent_child.parent_store import ParentStore
from src.rag.parent_child.registry import GenerationRegistryRecord
from src.rag.parent_child.retrieval import (
    ChildReranker,
    HybridRetrievalPolicy,
    ParentChildHybridRetriever,
)
from src.rag.parent_child.runtime_resources import (
    ChromaChildSearchChannel,
    GenerationResources,
    QueryEmbeddingProvider,
    SubjectBm25Router,
    SubjectBm25SearchChannel,
)


class GenerationRuntimeLoadError(RuntimeError):
    """A READY generation cannot be safely used by candidate retrieval."""


def compute_reranker_fingerprint(config: RagIndexConfig) -> str:
    """Fingerprint non-secret reranker identity and response contract."""

    reranker = config.reranker
    return sha256_bytes(
        canonical_json_bytes(
            {
                "base_url": reranker.base_url,
                "endpoint_path": reranker.endpoint_path,
                "model": reranker.model,
                "protocol": reranker.protocol,
                "provider": reranker.provider,
                "score_max": reranker.score_max,
                "score_min": reranker.score_min,
            }
        )
    )


def retrieval_policy_from_generation(
    config: RagIndexConfig,
    *,
    manifest_sha256: str,
    embedding_fingerprint: str,
    bm25_tokenizer_fingerprint: str,
) -> HybridRetrievalPolicy:
    retrieval = config.retrieval
    return HybridRetrievalPolicy(
        schema_version="hybrid_retrieval_policy_v1",
        generation_manifest_sha256=manifest_sha256,
        embedding_fingerprint=embedding_fingerprint,
        bm25_tokenizer_fingerprint=bm25_tokenizer_fingerprint,
        reranker_fingerprint=compute_reranker_fingerprint(config),
        vector_top_k=retrieval.vector_top_k,
        bm25_top_k=retrieval.bm25_top_k,
        vector_rrf_weight=retrieval.vector_weight,
        bm25_rrf_weight=retrieval.bm25_weight,
        rrf_k=retrieval.rrf_k,
        reranker_top_n=retrieval.reranker_top_n,
        unique_parent_top_k=retrieval.unique_parent_top_k,
        max_children_per_parent=retrieval.max_children_per_parent,
        max_parents_per_source=retrieval.max_parents_per_source,
        parent_support_lambda=retrieval.parent_support_lambda,
        full_parent_max_chars=retrieval.full_parent_max_chars,
        hit_window_chars_per_side=retrieval.hit_window_chars_per_side,
        multi_subject_per_subject_top_k=retrieval.multi_subject_per_subject_top_k,
        multi_subject_max_parents=retrieval.multi_subject_max_parents,
        subject_coverage_quota=retrieval.subject_coverage_quota,
    )


@dataclass(frozen=True, slots=True)
class LoadedGenerationRuntime:
    """Verified available subjects, resources, and retrieval fingerprint policy."""

    generation_id: str
    available_subjects: tuple[str, ...]
    resources: GenerationResources
    retrieval_policy: HybridRetrievalPolicy
    cross_branch_rrf_k: int
    judge_preview_max_chars: int

    def retriever(self) -> ParentChildHybridRetriever:
        return ParentChildHybridRetriever(
            policy=self.retrieval_policy,
            vector_search=self.resources.vector,
            bm25_search=self.resources.bm25,
            reranker=self.resources.reranker,
            parent_hydrator=self.resources.parents,
        )

    def close(self) -> None:
        self.resources.close()


def _validate_manifest_against_config(
    config: RagIndexConfig,
    *,
    manifest_sha256: str,
    generation_root: Path,
    generation_id: str,
) -> tuple[GenerationManifest, tuple[str, ...]]:
    manifest = validate_sealed_generation(
        config.storage.index_root,
        generation_id,
        expected_manifest_sha256=manifest_sha256,
        expected_marker_schema_version=config.storage.owner_marker_schema_version,
    )
    if manifest.collection_name != config.storage.collection_name:
        raise GenerationRuntimeLoadError("collection name differs from strict config")
    if manifest.embedding.fingerprint != compute_embedding_fingerprint(config):
        raise GenerationRuntimeLoadError("embedding fingerprint differs from config")
    if (
        manifest.embedding.dimension != config.embedding.expected_dimension
        or manifest.embedding.distance_metric != config.embedding.distance_metric
    ):
        raise GenerationRuntimeLoadError("embedding shape/metric differs from config")
    expected_tokenizer = compute_tokenizer_fingerprint(
        tokenizer_name=config.bm25.tokenizer,
        tokenizer_version=config.bm25.tokenizer_version,
        dictionary_sha256=config.bm25.dictionary_hash,
    )
    if manifest.bm25.tokenizer_fingerprint != expected_tokenizer:
        raise GenerationRuntimeLoadError("BM25 fingerprint differs from config")

    subject_manifest = read_strict_model(
        generation_root,
        "subject_manifest.json",
        SubjectManifest,
    )
    policy_manifest = read_strict_model(
        generation_root,
        "policy_manifest.json",
        PolicyManifestSet,
    )
    if subject_manifest.generation_id != generation_id:
        raise GenerationRuntimeLoadError("subject manifest generation mismatch")
    active_entries = tuple(
        entry for entry in subject_manifest.entries if entry.exclusion_state == "active"
    )
    manifest_policy_map = {
        entry.subject_id: entry.policy_id for entry in active_entries
    }
    if manifest_policy_map != config.subject_policy_map:
        raise GenerationRuntimeLoadError(
            "active subject/policy inventory differs from strict config"
        )
    configured_policy_ids = set(config.chunk_policies)
    sealed_policy_ids = {policy.policy_id for policy in policy_manifest.policies}
    if configured_policy_ids != sealed_policy_ids:
        raise GenerationRuntimeLoadError("sealed policy inventory differs from config")
    available_subjects = tuple(entry.subject_id for entry in active_entries)
    if available_subjects != tuple(sorted(available_subjects)):
        raise GenerationRuntimeLoadError("available subjects are not sorted")
    return manifest, available_subjects


def load_generation_runtime(
    *,
    config: RagIndexConfig,
    registry_record: GenerationRegistryRecord,
    query_embedding_provider: QueryEmbeddingProvider,
    reranker: ChildReranker,
    bm25_tokenizer: Callable[[str], Sequence[str]],
) -> LoadedGenerationRuntime:
    """Open exact READY resources; any mismatch closes partial resources and fails."""

    if registry_record.state != "READY" or registry_record.manifest_sha256 is None:
        raise GenerationRuntimeLoadError(
            "runtime resources require a READY registry row"
        )
    generation_id = registry_record.generation_id
    logical_generation_root = (
        config.storage.index_root / registry_record.directory_relative_path
    )
    if logical_generation_root.is_symlink():
        raise GenerationRuntimeLoadError("generation directory must not be a symlink")
    generation_root = logical_generation_root.resolve(strict=True)
    if not generation_root.is_relative_to(
        config.storage.index_root.resolve(strict=True)
    ):
        raise GenerationRuntimeLoadError("generation path escapes index_root")
    manifest, available_subjects = _validate_manifest_against_config(
        config,
        manifest_sha256=registry_record.manifest_sha256,
        generation_root=generation_root,
        generation_id=generation_id,
    )

    vector: ChromaChildSearchChannel | None = None
    parents: ParentStore | None = None
    try:
        vector = ChromaChildSearchChannel(
            persist_directory=(generation_root / "chroma_children").resolve(
                strict=True
            ),
            collection_name=manifest.collection_name,
            generation_id=generation_id,
            expected_dimension=manifest.embedding.dimension,
            distance_metric=manifest.embedding.distance_metric,
            query_embedding_provider=query_embedding_provider,
        )
        parents = ParentStore.open_readonly(
            generation_root,
            "parents.sqlite",
            expected_schema_version=config.storage.parent_store_schema_version,
            expected_generation_id=generation_id,
            busy_timeout_seconds=config.storage.parent_store_busy_timeout_seconds,
        )
        parents.verify_integrity()
        channels = {
            subject: SubjectBm25SearchChannel.load(
                generation_root=generation_root,
                manifest_relative_path=f"bm25/{subject}.manifest.json",
                manifest_schema_version="bm25_manifest_v1",
                generation_id=generation_id,
                subject=subject,
                tokenizer_fingerprint=manifest.bm25.tokenizer_fingerprint,
                tokenizer=bm25_tokenizer,
                child_lookup=vector,
            )
            for subject in available_subjects
        }
        resources = GenerationResources(
            generation_id=generation_id,
            manifest_fingerprint=registry_record.manifest_sha256,
            vector=vector,
            bm25=SubjectBm25Router(channels),
            reranker=reranker,
            parents=parents,
        )
        policy = retrieval_policy_from_generation(
            config,
            manifest_sha256=registry_record.manifest_sha256,
            embedding_fingerprint=manifest.embedding.fingerprint,
            bm25_tokenizer_fingerprint=manifest.bm25.tokenizer_fingerprint,
        )
        return LoadedGenerationRuntime(
            generation_id=generation_id,
            available_subjects=available_subjects,
            resources=resources,
            retrieval_policy=policy,
            cross_branch_rrf_k=config.retrieval.cross_branch_rrf_k,
            judge_preview_max_chars=config.retrieval.judge_preview_max_chars,
        )
    except Exception:
        if parents is not None:
            parents.close()
        if vector is not None:
            vector.close()
        raise


__all__ = [
    "GenerationRuntimeLoadError",
    "LoadedGenerationRuntime",
    "compute_reranker_fingerprint",
    "load_generation_runtime",
    "retrieval_policy_from_generation",
]
