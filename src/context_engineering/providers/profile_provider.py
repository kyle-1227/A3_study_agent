"""Context provider for already-available profile summaries."""

from __future__ import annotations

from typing import Any

from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.schema import ContextItem, ContextProviderError


class ProfileContextProvider:
    """Objectize profile context that is already present in state/messages."""

    name = "profile_provider"
    source_type = "profile"

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        if context.max_items_per_provider <= 0:
            return []
        try:
            content, metadata = _profile_summary_from_state(context.state)
            if not content:
                return []
            return [
                make_context_item(
                    source_type="profile",
                    title="learner_profile",
                    content=content,
                    priority=70,
                    scope="global",
                    lifetime="long_term",
                    compressible=True,
                    can_drop=True,
                    disclosure_level="summary",
                    confidence=_score(metadata.get("confidence")),
                    metadata=metadata,
                    max_content_chars=context.max_content_chars_per_item,
                )
            ]
        except ContextProviderError:
            raise
        except Exception as exc:
            raise ContextProviderError(
                provider=self.name,
                source_type=self.source_type,
                stage="collect",
                message=exc,
                original_exception_type=type(exc).__name__,
            ) from exc


def _profile_summary_from_state(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for key in ("profile_summary", "profile_context", "learner_profile_summary"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), {"profile_source": key}

    profile = state.get("profile") or state.get("user_profile")
    if not profile:
        return "", {}
    if not isinstance(profile, dict):
        raise ContextProviderError(
            provider=ProfileContextProvider.name,
            source_type=ProfileContextProvider.source_type,
            stage="decode_state",
            message="profile must be a dict when present",
            original_exception_type="TypeError",
        )
    for key in ("summary", "profile_summary", "content"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), {
                "profile_source": f"profile.{key}",
                "user_id": profile.get("user_id", ""),
                "confidence": profile.get("confidence"),
            }
    return "", {}


def _score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    score = float(value)
    if score < 0.0 or score > 1.0:
        return None
    return score
