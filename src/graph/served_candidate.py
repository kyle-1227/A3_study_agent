"""Strict production assembly for the active Parent--Child primary runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.config.evidence_orchestration_config import (
    load_evidence_orchestration_config,
    load_resource_evidence_profiles,
)
from src.config.rag_index_config import RagIndexConfig, load_rag_index_config
from src.graph.evidence_orchestration import EvidenceOrchestrationRuntime
from src.graph.parent_child_nodes import parent_child_graph_runtime_from_loaded
from src.learning_guidance.runtime import LearningGuidanceRuntime
from src.rag.parent_child.provider_clients import (
    StrictEmbeddingClient,
    StrictRerankerClient,
)
from src.rag.parent_child.runtime_loader import (
    LoadedPrimaryRuntime,
    load_primary_runtime,
)
from src.rag.parent_child.tokenizer import ConfiguredJiebaTokenizer


class ServedPrimaryRuntimeError(RuntimeError):
    """The active primary cannot be safely assembled for production serving."""


@dataclass(frozen=True, slots=True)
class ServedPrimaryRuntime:
    """Owned production resources and a graph-facing primary orchestration."""

    orchestration: EvidenceOrchestrationRuntime
    primary_revision: int
    primary_updated_at: datetime
    primary_config_fingerprint: str
    _loaded_primary: LoadedPrimaryRuntime
    _embedding_client: StrictEmbeddingClient
    _reranker_client: StrictRerankerClient

    def close(self) -> None:
        try:
            self._loaded_primary.close()
        finally:
            try:
                self._embedding_client.close()
            finally:
                self._reranker_client.close()


def _runtime_index_config(
    *,
    source: RagIndexConfig,
    index_root: Path,
) -> RagIndexConfig:
    """Revalidate the checked-in provider policy against the mounted index root."""

    try:
        resolved_root = index_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ServedPrimaryRuntimeError(
            "Parent--Child index root must be an existing non-symlink directory"
        ) from exc
    if not resolved_root.is_dir() or resolved_root.is_symlink():
        raise ServedPrimaryRuntimeError(
            "Parent--Child index root must be an existing non-symlink directory"
        )
    payload = source.model_dump(mode="python")
    storage = dict(payload["storage"])
    storage["index_root"] = resolved_root
    # Registry config remains parseable during the transition but is not opened,
    # read, or otherwise used by the primary serving path.
    storage["registry_path"] = resolved_root / "generation_registry.sqlite"
    payload["storage"] = storage
    return RagIndexConfig.model_validate(payload)


def _contained_project_file(project_root: Path, path: Path, *, name: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise TypeError(f"{name} must be an absolute Path")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(project_root):
        raise ServedPrimaryRuntimeError(f"{name} must be a project-contained file")
    return resolved


def load_served_primary_runtime(
    *,
    project_root: Path,
    learning_guidance: LearningGuidanceRuntime,
    index_config_path: Path,
    index_root: Path,
    policy_config_path: Path,
    profiles_config_path: Path,
) -> ServedPrimaryRuntime:
    """Load the sole verified primary and fail closed on any identity mismatch."""

    if not isinstance(project_root, Path):
        raise TypeError("project_root must be Path")
    root = project_root.resolve(strict=True)
    if not root.is_dir():
        raise ServedPrimaryRuntimeError("project_root must be a directory")
    if not isinstance(learning_guidance, LearningGuidanceRuntime):
        raise TypeError("learning_guidance must be LearningGuidanceRuntime")
    index_config = _contained_project_file(
        root, index_config_path, name="index_config_path"
    )
    policy_config = _contained_project_file(
        root, policy_config_path, name="policy_config_path"
    )
    profiles_config = _contained_project_file(
        root, profiles_config_path, name="profiles_config_path"
    )
    if not isinstance(index_root, Path) or not index_root.is_absolute():
        raise TypeError("index_root must be an absolute Path")

    config = _runtime_index_config(
        source=load_rag_index_config(index_config),
        index_root=index_root,
    )
    policy = load_evidence_orchestration_config(policy_config)
    profiles = load_resource_evidence_profiles(profiles_config)

    embedding: StrictEmbeddingClient | None = None
    reranker: StrictRerankerClient | None = None
    loaded: LoadedPrimaryRuntime | None = None
    try:
        embedding = StrictEmbeddingClient.production(config=config.embedding)
        reranker = StrictRerankerClient.production(config=config.reranker)
        loaded = load_primary_runtime(
            config=config,
            query_embedding_provider=embedding,
            reranker=reranker,
            bm25_tokenizer=ConfiguredJiebaTokenizer(config=config.bm25),
        )
        parent_child = parent_child_graph_runtime_from_loaded(loaded=loaded)
        orchestration = EvidenceOrchestrationRuntime(
            parent_child=parent_child,
            policy=policy,
            profiles=profiles,
            learning_guidance=learning_guidance,
            web_timeout_seconds=policy.web_timeout_seconds,
        )
        return ServedPrimaryRuntime(
            orchestration=orchestration,
            primary_revision=loaded.primary_revision,
            primary_updated_at=loaded.primary_updated_at,
            primary_config_fingerprint=loaded.primary_config_fingerprint,
            _loaded_primary=loaded,
            _embedding_client=embedding,
            _reranker_client=reranker,
        )
    except BaseException:
        if loaded is not None:
            loaded.close()
        if embedding is not None:
            embedding.close()
        if reranker is not None:
            reranker.close()
        raise


# Kept as an import-only compatibility alias while callers move to the explicit
# primary name. It carries no registry, manifest, READY, shadow, or rollback
# behavior.
ServedCandidateRuntime = ServedPrimaryRuntime
ServedCandidateRuntimeError = ServedPrimaryRuntimeError


__all__ = [
    "ServedCandidateRuntime",
    "ServedCandidateRuntimeError",
    "ServedPrimaryRuntime",
    "ServedPrimaryRuntimeError",
    "_runtime_index_config",
    "load_served_primary_runtime",
]
