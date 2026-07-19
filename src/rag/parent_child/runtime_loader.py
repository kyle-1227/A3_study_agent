"""Load only the verified mutable Parent--Child primary runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.config.rag_index_config import RagIndexConfig
from src.rag.parent_child._storage_io import canonical_json_bytes, sha256_bytes
from src.rag.parent_child.chroma_runtime_snapshot import (
    CHROMA_RUNTIME_OWNER_SCHEMA_VERSION,
    ChromaRuntimeSnapshot,
    ChromaRuntimeSnapshotError,
)
from src.rag.parent_child.parent_store import ParentStore
from src.rag.parent_child.primary_runtime import (
    PrimaryIndexError,
    PrimaryIndexMetadataV1,
    PrimaryIndexStateV1,
    primary_artifact_root,
    load_primary_metadata,
    load_primary_state,
    load_primary_validation,
    validate_primary_revision,
)
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


class PrimaryRuntimeLoadError(RuntimeError):
    """The active primary cannot be safely served."""


def compute_reranker_fingerprint(config: RagIndexConfig) -> str:
    """Fingerprint non-secret reranker identity and response contract."""

    reranker = config.reranker
    return sha256_bytes(
        canonical_json_bytes(
            {
                "base_url": reranker.base_url,
                "batch_recovery": reranker.batch_recovery.model_dump(mode="json"),
                "batch_size": reranker.batch_size,
                "endpoint_path": reranker.endpoint_path,
                "model": reranker.model,
                "protocol": reranker.protocol,
                "provider": reranker.provider,
                "provider_routing": (
                    None
                    if reranker.provider_routing is None
                    else {
                        "allow_fallbacks": reranker.provider_routing.allow_fallbacks,
                        "order": reranker.provider_routing.order,
                    }
                ),
                "retry": reranker.retry.model_dump(mode="json"),
                "score_max": reranker.score_max,
                "score_min": reranker.score_min,
                "timeout_seconds": reranker.timeout_seconds,
            }
        )
    )


def retrieval_policy_from_primary(
    config: RagIndexConfig,
    *,
    primary_revision: int,
    primary_config_fingerprint: str,
    embedding_fingerprint: str,
    bm25_tokenizer_fingerprint: str,
) -> HybridRetrievalPolicy:
    retrieval = config.retrieval
    return HybridRetrievalPolicy(
        schema_version="hybrid_retrieval_policy_v2",
        primary_revision=primary_revision,
        primary_config_fingerprint=primary_config_fingerprint,
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
class LoadedPrimaryRuntime:
    """Verified resources and immutable-in-process primary identity."""

    primary_revision: int
    primary_updated_at: datetime
    primary_config_fingerprint: str
    artifact_identity: str
    available_subjects: tuple[str, ...]
    resources: GenerationResources
    chroma_snapshot: ChromaRuntimeSnapshot
    retrieval_policy: HybridRetrievalPolicy
    cross_branch_rrf_k: int
    judge_preview_max_chars: int

    @property
    def generation_id(self) -> str:
        """Internal artifact identity retained for Parent--Child record checks."""

        return self.artifact_identity

    def retriever(self) -> ParentChildHybridRetriever:
        return ParentChildHybridRetriever(
            policy=self.retrieval_policy,
            vector_search=self.resources.vector,
            bm25_search=self.resources.bm25,
            reranker=self.resources.reranker,
            parent_hydrator=self.resources.parents,
        )

    def close(self) -> None:
        try:
            self.resources.close()
        finally:
            self.chroma_snapshot.close()


def _assert_primary_loaded(
    *,
    config: RagIndexConfig,
    index_root: Path,
) -> tuple[PrimaryIndexStateV1, PrimaryIndexMetadataV1, Path]:
    try:
        state = load_primary_state(index_root)
        metadata = load_primary_metadata(index_root, state=state)
        load_primary_validation(index_root, state=state, metadata=metadata)
        root = primary_artifact_root(index_root, state=state)
    except (FileNotFoundError, PrimaryIndexError, ValueError) as exc:
        raise PrimaryRuntimeLoadError(
            "active Parent--Child primary is invalid"
        ) from exc
    return state, metadata, root


def load_primary_runtime(
    *,
    config: RagIndexConfig,
    query_embedding_provider: QueryEmbeddingProvider,
    reranker: ChildReranker,
    bm25_tokenizer: Callable[[str], Sequence[str]],
) -> LoadedPrimaryRuntime:
    """Open exactly one verified primary; missing or corrupt state fails closed."""

    state, metadata, root = _assert_primary_loaded(
        config=config,
        index_root=config.storage.index_root,
    )
    vector: ChromaChildSearchChannel | None = None
    parents: ParentStore | None = None
    snapshot: ChromaRuntimeSnapshot | None = None
    try:
        try:
            snapshot = ChromaRuntimeSnapshot.create(
                index_root=config.storage.index_root,
                source_directory=root / metadata.chroma_directory_relative_path,
                owner_schema_version=CHROMA_RUNTIME_OWNER_SCHEMA_VERSION,
            )
            validate_primary_revision(
                config=config,
                artifact_root=root,
                metadata=metadata,
                chroma_snapshot=snapshot,
            )
        except (ChromaRuntimeSnapshotError, PrimaryIndexError, ValueError) as exc:
            raise PrimaryRuntimeLoadError(
                "active Parent--Child primary is invalid"
            ) from exc
        vector = ChromaChildSearchChannel(
            persist_directory=snapshot.persist_directory,
            collection_name=metadata.collection_name,
            generation_id=metadata.artifact_identity,
            expected_dimension=metadata.embedding_dimension,
            distance_metric=metadata.distance_metric,
            query_embedding_provider=query_embedding_provider,
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
        channels = {
            subject: SubjectBm25SearchChannel.load(
                generation_root=root,
                manifest_relative_path=(
                    f"{metadata.bm25_directory_relative_path}/{subject}.manifest.json"
                ),
                manifest_schema_version="bm25_manifest_v1",
                generation_id=metadata.artifact_identity,
                subject=subject,
                tokenizer_fingerprint=metadata.bm25_tokenizer_fingerprint,
                tokenizer=bm25_tokenizer,
                child_lookup=vector,
            )
            for subject in metadata.available_subjects
        }
        resources = GenerationResources(
            generation_id=metadata.artifact_identity,
            manifest_fingerprint=state.config_fingerprint,
            vector=vector,
            bm25=SubjectBm25Router(channels),
            reranker=reranker,
            parents=parents,
        )
        policy = retrieval_policy_from_primary(
            config,
            primary_revision=state.primary_revision,
            primary_config_fingerprint=state.config_fingerprint,
            embedding_fingerprint=metadata.embedding_fingerprint,
            bm25_tokenizer_fingerprint=metadata.bm25_tokenizer_fingerprint,
        )
        return LoadedPrimaryRuntime(
            primary_revision=state.primary_revision,
            primary_updated_at=state.updated_at_utc,
            primary_config_fingerprint=state.config_fingerprint,
            artifact_identity=metadata.artifact_identity,
            available_subjects=metadata.available_subjects,
            resources=resources,
            chroma_snapshot=snapshot,
            retrieval_policy=policy,
            cross_branch_rrf_k=config.retrieval.cross_branch_rrf_k,
            judge_preview_max_chars=config.retrieval.judge_preview_max_chars,
        )
    except BaseException:
        if parents is not None:
            parents.close()
        if vector is not None:
            vector.close()
        if snapshot is not None:
            snapshot.close()
        raise


__all__ = [
    "LoadedPrimaryRuntime",
    "PrimaryRuntimeLoadError",
    "compute_reranker_fingerprint",
    "load_primary_runtime",
    "retrieval_policy_from_primary",
]
