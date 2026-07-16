"""Strict production assembly for the active Parent-Child served runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.config.evidence_orchestration_config import (
    load_evidence_orchestration_config,
    load_resource_evidence_profiles,
)
from src.config.rag_index_config import RagIndexConfig, load_rag_index_config
from src.config.rag_rollout_config import load_rag_rollout_config
from src.graph.evidence_orchestration import EvidenceOrchestrationRuntime
from src.graph.parent_child_nodes import parent_child_graph_runtime_from_loaded
from src.learning_guidance.runtime import LearningGuidanceRuntime
from src.rag.parent_child.provider_clients import (
    StrictEmbeddingClient,
    StrictRerankerClient,
)
from src.rag.parent_child.registry import DeploymentSnapshot, GenerationRegistry
from src.rag.parent_child.runtime_loader import (
    LoadedGenerationRuntime,
    load_generation_runtime,
)
from src.rag.parent_child.tokenizer import ConfiguredJiebaTokenizer


class ServedCandidateRuntimeError(RuntimeError):
    """The explicitly selected Parent-Child generation cannot be safely served."""


@dataclass(frozen=True, slots=True)
class ServedCandidateRuntime:
    """Owned production resources plus the graph-facing orchestration runtime."""

    orchestration: EvidenceOrchestrationRuntime
    generation_manifest_fingerprint: str
    deployment_mode: Literal["active"]
    rollout_activation_enabled: bool
    rollout_shadow_enabled: bool
    _loaded_generation: LoadedGenerationRuntime
    _embedding_client: StrictEmbeddingClient
    _reranker_client: StrictRerankerClient

    def close(self) -> None:
        """Release generation and Provider transports exactly once per owner."""

        try:
            self._loaded_generation.close()
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
        raise ServedCandidateRuntimeError(
            "Parent-Child index root must be an existing non-symlink directory"
        ) from exc
    if not resolved_root.is_dir() or resolved_root.is_symlink():
        raise ServedCandidateRuntimeError(
            "Parent-Child index root must be an existing non-symlink directory"
        )
    payload = source.model_dump(mode="python")
    storage = dict(payload["storage"])
    storage["index_root"] = resolved_root
    storage["registry_path"] = resolved_root / "generation_registry.sqlite"
    payload["storage"] = storage
    return RagIndexConfig.model_validate(payload)


def _validate_production_deployment(
    *,
    generation_id: str,
    deployment: DeploymentSnapshot,
) -> None:
    """Require one active primary with no shadow or duplicate previous pointer."""
    if deployment.primary_generation_id != generation_id:
        raise ServedCandidateRuntimeError(
            "served generation must match the active registry primary"
        )
    if deployment.shadow_generation_id is not None:
        raise ServedCandidateRuntimeError(
            "active production serving requires an empty shadow pointer"
        )
    if deployment.previous_generation_id == generation_id:
        raise ServedCandidateRuntimeError(
            "active and previous generation pointers must be distinct"
        )


def load_served_candidate_runtime(
    *,
    project_root: Path,
    generation_id: str,
    learning_guidance: LearningGuidanceRuntime,
    index_config_path: Path,
    index_root: Path,
    policy_config_path: Path,
    profiles_config_path: Path,
    rollout_config_path: Path,
) -> ServedCandidateRuntime:
    """Load the exact READY generation selected as the active production primary.

    The generation id is a required caller input. The registry must already name it
    as primary, and no shadow pointer may exist.
    """

    if not isinstance(project_root, Path):
        raise TypeError("project_root must be Path")
    root = project_root.resolve(strict=True)
    if not root.is_dir():
        raise ServedCandidateRuntimeError("project_root must be a directory")
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or generation_id != generation_id.strip()
    ):
        raise ServedCandidateRuntimeError(
            "generation_id must be an explicit non-blank stripped identifier"
        )
    if not isinstance(learning_guidance, LearningGuidanceRuntime):
        raise TypeError("learning_guidance must be LearningGuidanceRuntime")
    for field_name, path in (
        ("index_config_path", index_config_path),
        ("policy_config_path", policy_config_path),
        ("profiles_config_path", profiles_config_path),
        ("rollout_config_path", rollout_config_path),
    ):
        if not isinstance(path, Path) or not path.is_absolute():
            raise TypeError(f"{field_name} must be an absolute Path")
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise ServedCandidateRuntimeError(
                f"{field_name} must be a project-contained file"
            )
    if not isinstance(index_root, Path) or not index_root.is_absolute():
        raise TypeError("index_root must be an absolute Path")

    rollout = load_rag_rollout_config(rollout_config_path)
    if rollout.activation_enabled is not True or rollout.shadow_enabled is not False:
        raise ServedCandidateRuntimeError(
            "production serving requires activation enabled and shadow disabled"
        )

    source_config = load_rag_index_config(index_config_path)
    config = _runtime_index_config(
        source=source_config,
        index_root=index_root,
    )
    policy = load_evidence_orchestration_config(policy_config_path)
    profiles = load_resource_evidence_profiles(profiles_config_path)

    registry = GenerationRegistry.open(
        config.storage.resolved_registry_path(),
        index_root=config.storage.index_root,
        expected_schema_version=config.storage.registry_schema_version,
        marker_schema_version=config.storage.owner_marker_schema_version,
        busy_timeout_seconds=config.storage.registry_busy_timeout_seconds,
    )
    try:
        deployment = registry.deployment()
        _validate_production_deployment(
            generation_id=generation_id,
            deployment=deployment,
        )
        record = registry.get_generation(generation_id)
        if record.state != "READY" or record.manifest_sha256 is None:
            raise ServedCandidateRuntimeError(
                "served candidate generation must be sealed and READY"
            )
    finally:
        registry.close()

    embedding: StrictEmbeddingClient | None = None
    reranker: StrictRerankerClient | None = None
    loaded: LoadedGenerationRuntime | None = None
    try:
        embedding = StrictEmbeddingClient.production(config=config.embedding)
        reranker = StrictRerankerClient.production(config=config.reranker)
        tokenizer = ConfiguredJiebaTokenizer(config=config.bm25)
        loaded = load_generation_runtime(
            config=config,
            registry_record=record,
            query_embedding_provider=embedding,
            reranker=reranker,
            bm25_tokenizer=tokenizer,
        )
        parent_child = parent_child_graph_runtime_from_loaded(loaded=loaded)
        orchestration = EvidenceOrchestrationRuntime(
            parent_child=parent_child,
            policy=policy,
            profiles=profiles,
            learning_guidance=learning_guidance,
            web_timeout_seconds=policy.web_timeout_seconds,
        )
        return ServedCandidateRuntime(
            orchestration=orchestration,
            generation_manifest_fingerprint=record.manifest_sha256,
            deployment_mode="active",
            rollout_activation_enabled=rollout.activation_enabled,
            rollout_shadow_enabled=rollout.shadow_enabled,
            _loaded_generation=loaded,
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


__all__ = [
    "ServedCandidateRuntime",
    "ServedCandidateRuntimeError",
    "load_served_candidate_runtime",
]
