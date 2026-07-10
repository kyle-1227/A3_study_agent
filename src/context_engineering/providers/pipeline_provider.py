"""Read-only provider for compact cross-node influence entries."""

from __future__ import annotations

from typing import Any, Mapping, TypeGuard

from src.context_engineering.influence import (
    INFLUENCE_ID_PREFIX,
    INFLUENCE_LEDGER_SCHEMA_VERSION,
    influence_kind_is_injectable,
)
from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.schema import ContextItem
from src.context_engineering.workspace import sanitize_workspace_text


class PipelineContextProvider:
    """Expose eligible pipeline continuity summaries as ContextItems."""

    name = "pipeline_provider"
    source_type = "pipeline"

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        if context.max_items_per_provider <= 0:
            return []
        ledger = context.state.get("context_influence_ledger")
        if not isinstance(ledger, Mapping):
            return []
        version = ledger.get("schema_version")
        if version not in (None, INFLUENCE_LEDGER_SCHEMA_VERSION):
            return []
        entries_by_id = ledger.get("entries_by_id")
        ordered_ids = ledger.get("ordered_ids")
        if not isinstance(entries_by_id, Mapping) or not isinstance(ordered_ids, list):
            return []

        items: list[ContextItem] = []
        seen_ids: set[str] = set()
        for influence_id in reversed(ordered_ids):
            if len(items) >= context.max_items_per_provider:
                break
            if not isinstance(influence_id, str) or influence_id in seen_ids:
                continue
            entry = entries_by_id.get(influence_id)
            if not _valid_injectable_entry(entry, influence_id):
                continue
            seen_ids.add(influence_id)
            entry_mapping = entry
            metadata = entry_mapping.get("metadata")
            safe_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
            request_id = str(entry_mapping.get("request_id") or "")
            thread_id = str(entry_mapping.get("thread_id") or "")
            safe_metadata.update(
                {
                    "influence_id": influence_id,
                    "influence_kind": str(entry_mapping.get("kind") or ""),
                    "source_node": str(entry_mapping.get("source_node") or ""),
                    "target_stage": str(entry_mapping.get("target_stage") or ""),
                    "request_id": request_id,
                    "thread_id": thread_id,
                    "created_at": str(entry_mapping.get("created_at") or ""),
                    "content_fingerprint": str(
                        entry_mapping.get("content_fingerprint") or ""
                    ),
                    "purpose": "pipeline_continuity",
                }
            )
            safe_preview = sanitize_workspace_text(
                entry_mapping.get("preview"),
                max_chars=context.max_content_chars_per_item,
                fallback="",
            )
            if not safe_preview:
                continue
            items.append(
                make_context_item(
                    source_type="pipeline",
                    title=str(entry_mapping.get("title") or "pipeline_context"),
                    content=safe_preview,
                    priority=_bounded_priority(entry_mapping.get("priority")),
                    scope="session",
                    lifetime="session",
                    compressible=True,
                    can_drop=True,
                    disclosure_level="summary",
                    relevance_score=_relevance(
                        request_id=request_id,
                        thread_id=thread_id,
                        context=context,
                    ),
                    item_id=influence_id,
                    metadata=safe_metadata,
                    max_content_chars=context.max_content_chars_per_item,
                )
            )
        return items


def _valid_injectable_entry(
    entry: object, influence_id: str
) -> TypeGuard[Mapping[str, Any]]:
    if not isinstance(entry, Mapping):
        return False
    if str(entry.get("influence_id") or "") != influence_id:
        return False
    if not influence_id.startswith(f"{INFLUENCE_ID_PREFIX}:"):
        return False
    if not entry.get("injectable"):
        return False
    if not influence_kind_is_injectable(entry.get("kind")):
        return False
    preview = entry.get("preview")
    return isinstance(preview, str) and bool(preview.strip())


def _bounded_priority(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 50
    return max(0, min(value, 100))


def _relevance(
    *,
    request_id: str,
    thread_id: str,
    context: ProviderContext,
) -> float:
    if request_id and context.request_id and request_id == context.request_id:
        return 0.9
    if thread_id and context.thread_id and thread_id == context.thread_id:
        return 0.68
    if not thread_id or not context.thread_id:
        return 0.3
    return 0.05
