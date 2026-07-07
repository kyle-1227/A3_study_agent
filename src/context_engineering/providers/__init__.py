"""ContextProvider implementations and CE provider supply APIs."""

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
    get_registered_provider_sources,
)
from src.context_engineering.providers.rules_provider import RulesContextProvider
from src.context_engineering.providers.supply import (
    ContextCollectionResult,
    ProviderSupplyPlan,
    collect_context_for_policy,
    emit_context_items_collected_for_supply,
    emit_context_provider_supply,
    emit_context_provider_supply_plan,
    emit_provider_errors,
    plan_provider_supply,
)
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
    "ProviderSupplyPlan",
    "RulesContextProvider",
    "TrajectoryContextProvider",
    "ContextCollectionResult",
    "collect_context_for_policy",
    "collect_context_items",
    "collect_context_items_by_source",
    "emit_context_items_collected_for_supply",
    "emit_context_items_shadow",
    "emit_context_provider_supply",
    "emit_context_provider_supply_plan",
    "emit_provider_errors",
    "get_context_provider_settings",
    "get_default_providers",
    "get_registered_provider_sources",
    "plan_provider_supply",
]
