"""Explicit production composition for the four learning-guidance adapters."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import hashlib
import json
from pathlib import Path

from src.config.learning_guidance_config import (
    LearningGuidanceConfigV1,
    load_learning_guidance_config,
)
from src.learning_guidance.adapters.history import (
    HISTORY_ADAPTER_VERSION,
    HistorySnapshotAdapterV1,
)
from src.learning_guidance.adapters.path import (
    PATH_ENGINE_VERSION,
    LearnerPathEngineV1,
)
from src.learning_guidance.adapters.profile import (
    PROFILE_ADAPTER_VERSION,
    ProfileSnapshotAdapterV1,
)
from src.learning_guidance.adapters.recommendation import (
    RECOMMENDATION_ENGINE_VERSION,
    ResourceRecommendationEngineV1,
)
from src.learning_guidance.contracts import (
    LearnerHistorySnapshotV1,
    LearnerPathEngineRequestV1,
    LearnerPathPlanV1,
    LearnerProfileSnapshotV1,
    ResourceRecommendationEngineRequestV1,
    ResourceRecommendationEngineResultV1,
    build_learner_path_provider_policy_fingerprint,
)
from src.learning_guidance.knowledge_graph import (
    KnowledgeGraphV1,
    load_knowledge_graph,
)
from src.learning_guidance.runtime import LearningGuidanceRuntime
from src.memory.storage import SQLiteMemoryStore
from src.profile.storage import SQLiteProfileStore


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_adapter_versions(config: LearningGuidanceConfigV1) -> None:
    configured = (
        config.profile_adapter_version,
        config.history_adapter_version,
        config.path_engine_version,
        config.recommendation_engine_version,
    )
    implemented = (
        PROFILE_ADAPTER_VERSION,
        HISTORY_ADAPTER_VERSION,
        PATH_ENGINE_VERSION,
        RECOMMENDATION_ENGINE_VERSION,
    )
    if configured != implemented:
        raise RuntimeError(
            "learning-guidance configured adapter versions are not implemented"
        )


def _runtime_fingerprint(
    *,
    config: LearningGuidanceConfigV1,
    knowledge_graph: KnowledgeGraphV1,
) -> str:
    projection_fingerprint = build_learner_path_provider_policy_fingerprint(
        max_steps=config.provider_projection_max_steps,
        max_chars=config.provider_projection_max_chars,
    )
    return _canonical_digest(
        {
            "schema_version": "learning_guidance_runtime_v1",
            "policy_fingerprint": config.policy_fingerprint,
            "knowledge_graph_fingerprint": knowledge_graph.artifact_fingerprint,
            "adapter_versions": {
                "profile": config.profile_adapter_version,
                "history": config.history_adapter_version,
                "path": config.path_engine_version,
                "recommendation": config.recommendation_engine_version,
            },
            "provider_projection_policy_fingerprint": projection_fingerprint,
            "contracts": {
                "profile": LearnerProfileSnapshotV1.model_json_schema(),
                "history": LearnerHistorySnapshotV1.model_json_schema(),
                "path_request": LearnerPathEngineRequestV1.model_json_schema(),
                "path_plan": LearnerPathPlanV1.model_json_schema(),
                "recommendation_request": (
                    ResourceRecommendationEngineRequestV1.model_json_schema()
                ),
                "recommendation_result": (
                    ResourceRecommendationEngineResultV1.model_json_schema()
                ),
            },
        }
    )


def build_learning_guidance_runtime(
    *,
    config: LearningGuidanceConfigV1,
    knowledge_graph: KnowledgeGraphV1,
    profile_db_path: Path,
    memory_db_path: Path,
    clock: Callable[[], datetime],
) -> LearningGuidanceRuntime:
    """Build all real adapters from explicit validated dependencies."""

    if not isinstance(config, LearningGuidanceConfigV1):
        raise TypeError("config must be LearningGuidanceConfigV1")
    if not isinstance(knowledge_graph, KnowledgeGraphV1):
        raise TypeError("knowledge_graph must be KnowledgeGraphV1")
    if not isinstance(profile_db_path, Path):
        raise TypeError("profile_db_path must be pathlib.Path")
    if not isinstance(memory_db_path, Path):
        raise TypeError("memory_db_path must be pathlib.Path")
    if not callable(clock):
        raise TypeError("clock must be callable")
    _validate_adapter_versions(config)

    profile_adapter = ProfileSnapshotAdapterV1(
        store=SQLiteProfileStore(profile_db_path),
        knowledge_graph=knowledge_graph,
    )
    history_adapter = HistorySnapshotAdapterV1(
        store=SQLiteMemoryStore(memory_db_path),
        knowledge_graph=knowledge_graph,
        history_limit=config.history_limit,
    )
    path_engine = LearnerPathEngineV1(
        knowledge_graph=knowledge_graph,
        policy=config.path_policy,
        clock=clock,
    )
    recommendation_engine = ResourceRecommendationEngineV1(
        knowledge_graph=knowledge_graph,
        policy=config.recommendation_policy,
        clock=clock,
    )
    return LearningGuidanceRuntime(
        runtime_fingerprint=_runtime_fingerprint(
            config=config,
            knowledge_graph=knowledge_graph,
        ),
        knowledge_graph=knowledge_graph,
        provider_projection_max_steps=config.provider_projection_max_steps,
        provider_projection_max_chars=config.provider_projection_max_chars,
        load_profile=profile_adapter.load,
        load_history=history_adapter.load,
        plan_learning_path=path_engine.plan,
        recommend_resources=recommendation_engine.recommend,
    )


def resolve_knowledge_graph_path(
    *,
    config: LearningGuidanceConfigV1,
    project_root: Path,
) -> Path:
    """Resolve the configured relative artifact inside one explicit root."""

    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path")
    if config.knowledge_graph_path.is_absolute():
        raise ValueError("knowledge_graph_path must be relative to project_root")
    root = project_root.resolve()
    resolved = (root / config.knowledge_graph_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("knowledge_graph_path must remain inside project_root")
    return resolved


def load_learning_guidance_runtime(
    *,
    config_path: Path,
    project_root: Path,
    profile_db_path: Path,
    memory_db_path: Path,
    clock: Callable[[], datetime],
) -> LearningGuidanceRuntime:
    """Load policy and KG before constructing any store-backed adapter."""

    config = load_learning_guidance_config(config_path)
    knowledge_graph = load_knowledge_graph(
        resolve_knowledge_graph_path(config=config, project_root=project_root)
    )
    return build_learning_guidance_runtime(
        config=config,
        knowledge_graph=knowledge_graph,
        profile_db_path=profile_db_path,
        memory_db_path=memory_db_path,
        clock=clock,
    )


__all__ = [
    "build_learning_guidance_runtime",
    "load_learning_guidance_runtime",
    "resolve_knowledge_graph_path",
]
