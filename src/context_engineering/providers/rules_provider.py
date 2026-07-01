"""Context provider for explicit runtime rules already in state."""

from __future__ import annotations

from typing import Any

from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.schema import ContextItem, ContextProviderError


class RulesContextProvider:
    """Objectize rule/constraint summaries without exposing schemas."""

    name = "rules_provider"
    source_type = "rules"

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        if context.max_items_per_provider <= 0:
            return []
        try:
            rules = _existing_rules(context.state, limit=context.max_items_per_provider)
            return [
                make_context_item(
                    source_type="rules",
                    title=title,
                    content=content,
                    priority=95,
                    scope="node",
                    lifetime="turn",
                    compressible=False,
                    can_drop=False,
                    disclosure_level="summary",
                    metadata={"rule_source": source, "rule_index": index},
                    max_content_chars=context.max_content_chars_per_item,
                )
                for index, (source, title, content) in enumerate(rules)
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


def _existing_rules(
    state: dict[str, Any],
    *,
    limit: int,
) -> list[tuple[str, str, str]]:
    rules: list[tuple[str, str, str]] = []
    for key in ("context_rules", "node_rules", "runtime_rules"):
        value = state.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, str):
            rules.append((key, key, value))
            if len(rules) >= limit:
                return rules
            continue
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, str):
                    rules.append((key, f"{key}_{index}", item))
                elif isinstance(item, dict):
                    content = str(
                        item.get("summary")
                        or item.get("content")
                        or item.get("rule")
                        or ""
                    )
                    if content:
                        rules.append(
                            (
                                key,
                                str(item.get("title") or f"{key}_{index}"),
                                content,
                            )
                        )
                else:
                    raise ContextProviderError(
                        provider=RulesContextProvider.name,
                        source_type=RulesContextProvider.source_type,
                        stage="decode_state",
                        message=f"{key} list item must be str or dict",
                        original_exception_type="TypeError",
                    )
                if len(rules) >= limit:
                    return rules
            continue
        raise ContextProviderError(
            provider=RulesContextProvider.name,
            source_type=RulesContextProvider.source_type,
            stage="decode_state",
            message=f"{key} must be str or list",
            original_exception_type="TypeError",
        )
    return rules
