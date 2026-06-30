"""ContextProvider implementations for Phase 2 shadow objectization."""

from src.context_engineering.providers.artifact_provider import ArtifactContextProvider
from src.context_engineering.providers.base import ContextProvider, ProviderContext
from src.context_engineering.providers.curriculum_provider import (
    CurriculumContextProvider,
)
from src.context_engineering.providers.evidence_provider import EvidenceContextProvider
from src.context_engineering.providers.memory_provider import MemoryContextProvider
from src.context_engineering.providers.message_provider import MessageContextProvider
from src.context_engineering.providers.profile_provider import ProfileContextProvider
from src.context_engineering.providers.registry import (
    ContextProviderSettings,
    collect_context_items,
    collect_context_items_by_source,
    emit_context_items_shadow,
    get_context_provider_settings,
    get_default_providers,
)
from src.context_engineering.providers.rules_provider import RulesContextProvider
from src.context_engineering.providers.trajectory_provider import (
    TrajectoryContextProvider,
)

__all__ = [
    "ArtifactContextProvider",
    "ContextProvider",
    "ContextProviderSettings",
    "CurriculumContextProvider",
    "EvidenceContextProvider",
    "MemoryContextProvider",
    "MessageContextProvider",
    "ProfileContextProvider",
    "ProviderContext",
    "RulesContextProvider",
    "TrajectoryContextProvider",
    "collect_context_items",
    "collect_context_items_by_source",
    "emit_context_items_shadow",
    "get_context_provider_settings",
    "get_default_providers",
]
