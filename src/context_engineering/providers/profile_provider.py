"""Context provider for already-available profile summaries."""

from __future__ import annotations

from typing import Any

from src.context_engineering.itemizer import make_context_item, stable_item_id
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
            identity_metadata = _profile_identity_metadata(
                metadata,
                state=context.state,
                thread_id=context.thread_id,
            )
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
                    item_id=stable_item_id(
                        source_type="profile",
                        title=(
                            "learner_profile:"
                            f"{identity_metadata.get('user_id') or metadata.get('profile_source')}"
                        ),
                    ),
                    metadata={
                        **metadata,
                        **identity_metadata,
                        "purpose": "personalization",
                    },
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
    for key in (
        "profile_summary",
        "profile_context",
        "learner_profile_summary",
        "learner_profile",
        "preferences",
        "weaknesses",
        "strengths",
    ):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), {"profile_source": key}
        if isinstance(value, (dict, list)) and value:
            metadata: dict[str, Any] = {"profile_source": key}
            if isinstance(value, dict):
                metadata.update(
                    {
                        "user_id": value.get("user_id", ""),
                        "confidence": value.get("confidence"),
                    }
                )
            return str(value), metadata

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


def profile_summary_from_state(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return compact profile content using the provider's state contract."""
    return _profile_summary_from_state(state)


def _score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    score = float(value)
    if score < 0.0 or score > 1.0:
        return None
    return score


def _profile_identity_metadata(
    metadata: dict[str, Any],
    *,
    state: dict[str, Any],
    thread_id: str | None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    metadata_user_id = metadata.get("user_id")
    state_user_id = state.get("user_id")
    for value in (metadata_user_id, state_user_id):
        if isinstance(value, str) and value.strip():
            result["user_id"] = value.strip()
            break
    if isinstance(thread_id, str) and thread_id.strip():
        result["thread_id"] = thread_id.strip()
    return result
