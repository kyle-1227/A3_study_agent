"""Registry and shadow collection for ContextProvider implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence, cast

from src.config import get_setting
from src.context_engineering.providers.artifact_provider import ArtifactContextProvider
from src.context_engineering.providers.base import ContextProvider, ProviderContext
from src.context_engineering.providers.curriculum_provider import (
    CurriculumContextProvider,
)
from src.context_engineering.providers.evidence_provider import EvidenceContextProvider
from src.context_engineering.providers.memory_provider import MemoryContextProvider
from src.context_engineering.providers.message_provider import MessageContextProvider
from src.context_engineering.providers.profile_provider import ProfileContextProvider
from src.context_engineering.providers.rules_provider import RulesContextProvider
from src.context_engineering.providers.trajectory_provider import (
    TrajectoryContextProvider,
)
from src.context_engineering.schema import (
    ContextConfigError,
    ContextItem,
    ContextProviderError,
    ContextSourceType,
)
from src.context_engineering.trace import (
    emit_context_items_collected,
    emit_context_provider_error,
)


@dataclass(frozen=True)
class ContextProviderSettings:
    """Explicit settings for Phase 2 provider shadow collection."""

    enabled: bool
    shadow_mode: bool
    strict: bool
    enabled_sources: tuple[ContextSourceType, ...]
    max_items_per_provider: int
    max_content_chars_per_item: int
    trace_top_items: int


_PROVIDER_CLASSES = (
    MessageContextProvider,
    MemoryContextProvider,
    EvidenceContextProvider,
    ProfileContextProvider,
    RulesContextProvider,
    ArtifactContextProvider,
    TrajectoryContextProvider,
    CurriculumContextProvider,
)

_ALLOWED_SOURCES = {
    "message",
    "memory",
    "evidence",
    "artifact",
    "profile",
    "trajectory",
    "rules",
    "curriculum",
    "unknown",
}


def get_context_provider_settings() -> ContextProviderSettings:
    """Read explicit provider settings from context_engineering.providers."""
    context_config = get_setting("context_engineering")
    if not isinstance(context_config, dict):
        raise ContextConfigError(
            "context_engineering_missing",
            "context_engineering config is required",
        )
    if context_config.get("enabled") is False:
        return ContextProviderSettings(
            enabled=False,
            shadow_mode=False,
            strict=False,
            enabled_sources=(),
            max_items_per_provider=0,
            max_content_chars_per_item=0,
            trace_top_items=0,
        )

    providers = context_config.get("providers")
    if not isinstance(providers, dict):
        raise ContextConfigError(
            "context_providers_missing",
            "context_engineering.providers config is required",
        )

    enabled = _required_bool(providers, "enabled")
    shadow_mode = _required_bool(providers, "shadow_mode")
    strict = _required_bool(providers, "strict")
    return ContextProviderSettings(
        enabled=enabled,
        shadow_mode=shadow_mode,
        strict=strict,
        enabled_sources=_required_sources(providers),
        max_items_per_provider=_required_positive_int(
            providers,
            "max_items_per_provider",
        ),
        max_content_chars_per_item=_required_positive_int(
            providers,
            "max_content_chars_per_item",
        ),
        trace_top_items=_required_non_negative_int(providers, "trace_top_items"),
    )


def get_default_providers(
    settings: ContextProviderSettings | None = None,
) -> list[ContextProvider]:
    """Return configured default providers without external side effects."""
    settings = settings or get_context_provider_settings()
    if not settings.enabled:
        return []
    enabled_sources = set(settings.enabled_sources)
    providers: list[ContextProvider] = []
    for provider_cls in _PROVIDER_CLASSES:
        provider = provider_cls()
        if provider.source_type in enabled_sources:
            providers.append(cast(ContextProvider, provider))
    return providers


def collect_context_items(
    context: ProviderContext,
    *,
    providers: Sequence[ContextProvider] | None = None,
    settings: ContextProviderSettings | None = None,
) -> list[ContextItem]:
    """Collect context items without emitting trace events."""
    settings = settings or get_context_provider_settings()
    items, errors, _provider_count = _collect_with_errors(
        context,
        providers=providers,
        settings=settings,
    )
    if errors and settings.strict:
        raise errors[0]
    return items


def collect_context_items_by_source(
    context: ProviderContext,
    *,
    providers: Sequence[ContextProvider] | None = None,
    settings: ContextProviderSettings | None = None,
) -> dict[str, list[ContextItem]]:
    """Collect items grouped by source_type."""
    grouped: dict[str, list[ContextItem]] = {}
    for item in collect_context_items(
        context,
        providers=providers,
        settings=settings,
    ):
        grouped.setdefault(item.source_type, []).append(item)
    return grouped


def emit_context_items_shadow(
    logger: logging.Logger,
    *,
    node_name: str,
    llm_node: str | None,
    messages: list[Any],
    state: dict[str, Any] | None,
) -> list[ContextItem]:
    """Collect ContextItems in shadow mode and emit safe trace/SSE events."""
    settings = get_context_provider_settings()
    if not settings.enabled or not settings.shadow_mode:
        return []

    state = state or {}
    user_query, current_user_message_index = _user_query_from_messages(messages)
    context = ProviderContext(
        node_name=node_name,
        llm_node=llm_node,
        user_query=user_query,
        current_user_message_index=current_user_message_index,
        state=state,
        messages=list(messages or []),
        request_id=_optional_string(state.get("request_id")),
        thread_id=_optional_string(state.get("thread_id")),
        max_items_per_provider=settings.max_items_per_provider,
        max_content_chars_per_item=settings.max_content_chars_per_item,
    )
    items, errors, provider_count = _collect_with_errors(
        context,
        providers=None,
        settings=settings,
    )
    if errors and settings.strict:
        raise errors[0]
    for error in errors:
        emit_context_provider_error(
            logger,
            error=error,
            node_name=node_name,
            llm_node=llm_node or "",
            state=state,
        )
    emit_context_items_collected(
        logger,
        node_name=node_name,
        llm_node=llm_node or "",
        provider_count=provider_count,
        items=items,
        trace_top_items=settings.trace_top_items,
        state=state,
    )
    return items


def _collect_with_errors(
    context: ProviderContext,
    *,
    providers: Sequence[ContextProvider] | None,
    settings: ContextProviderSettings,
) -> tuple[list[ContextItem], list[ContextProviderError], int]:
    if not settings.enabled:
        return [], [], 0
    selected = (
        list(providers) if providers is not None else get_default_providers(settings)
    )
    enabled_sources = set(settings.enabled_sources)
    items: list[ContextItem] = []
    errors: list[ContextProviderError] = []
    provider_count = 0
    for provider in selected:
        if provider.source_type not in enabled_sources:
            continue
        provider_count += 1
        try:
            collected = provider.collect(context)
            if not isinstance(collected, list):
                raise ContextProviderError(
                    provider=provider.name,
                    source_type=provider.source_type,
                    stage="collect",
                    message="provider returned non-list result",
                    original_exception_type="TypeError",
                )
            for item in collected[: settings.max_items_per_provider]:
                if not isinstance(item, ContextItem):
                    raise ContextProviderError(
                        provider=provider.name,
                        source_type=provider.source_type,
                        stage="collect",
                        message="provider returned non-ContextItem item",
                        original_exception_type="TypeError",
                    )
                items.append(item)
        except ContextProviderError as exc:
            errors.append(exc)
            if settings.strict:
                break
        except Exception as exc:
            errors.append(
                ContextProviderError(
                    provider=provider.name,
                    source_type=provider.source_type,
                    stage="collect",
                    message=exc,
                    original_exception_type=type(exc).__name__,
                )
            )
            if settings.strict:
                break
    return items, errors, provider_count


def _required_bool(values: dict[str, Any], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ContextConfigError(
            f"context_providers_{key}_invalid",
            f"context_engineering.providers.{key} must be a boolean",
        )
    return value


def _required_positive_int(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContextConfigError(
            f"context_providers_{key}_invalid",
            f"context_engineering.providers.{key} must be a positive integer",
        )
    return value


def _required_non_negative_int(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContextConfigError(
            f"context_providers_{key}_invalid",
            f"context_engineering.providers.{key} must be a non-negative integer",
        )
    return value


def _required_sources(values: dict[str, Any]) -> tuple[ContextSourceType, ...]:
    raw = values.get("enabled_sources")
    if not isinstance(raw, list) or not raw:
        raise ContextConfigError(
            "context_providers_enabled_sources_invalid",
            "context_engineering.providers.enabled_sources must be a non-empty list",
        )
    sources: list[ContextSourceType] = []
    for item in raw:
        source = str(item or "").strip()
        if source not in _ALLOWED_SOURCES:
            raise ContextConfigError(
                "context_providers_enabled_sources_invalid",
                f"unknown context provider source: {source}",
            )
        sources.append(cast(ContextSourceType, source))
    return tuple(sources)


def _user_query_from_messages(messages: list[Any]) -> tuple[str, int | None]:
    from src.context_engineering.tokenizer import message_content_to_text

    for index in range(len(messages or []) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict):
            if str(message.get("role") or "").lower() == "user":
                return message_content_to_text(message).strip(), index
            continue
        class_name = type(message).__name__.lower()
        if "human" in class_name:
            return message_content_to_text(message).strip(), index
    return "", None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
