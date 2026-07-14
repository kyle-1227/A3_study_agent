"""Context provider for already-retrieved memory state."""

from __future__ import annotations

from typing import Any

from src.context_engineering.itemizer import make_context_item, stable_item_id
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.schema import ContextItem, ContextProviderError

_MEMORY_CONTENT_CHAR_CAPS = {
    "selected_memory": 200,
    "conversation_summary": 400,
    "memory_summary": 400,
    "memory_summaries": 400,
    "episodic_memory_results": 200,
    "semantic_memory_results": 250,
}


class MemoryContextProvider:
    """Objectize memory results already present in graph state."""

    name = "memory_provider"
    source_type = "memory"

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        if context.max_items_per_provider <= 0:
            return []
        if context.state.get("memory_use_policy") in {"ignore", "ask_user"}:
            return []
        try:
            results = _existing_memory_results(
                context.state,
                limit=context.max_items_per_provider,
            )
            return [
                _memory_result_to_item(
                    result,
                    source_bucket=bucket,
                    index=index,
                    thread_id=context.thread_id,
                    user_id=_explicit_user_id(context.state),
                    max_content_chars=context.max_content_chars_per_item,
                )
                for index, (bucket, result) in enumerate(results)
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


def _existing_memory_results(
    state: dict[str, Any],
    *,
    limit: int,
) -> list[tuple[str, dict[str, Any]]]:
    buckets: list[list[tuple[str, dict[str, Any]]]] = [[], [], [], []]
    selected_memory = state.get("selected_memory")
    if isinstance(selected_memory, dict):
        buckets[0].append(("selected_memory", selected_memory))
    elif isinstance(selected_memory, list):
        for item in selected_memory:
            if isinstance(item, dict):
                buckets[0].append(("selected_memory", item))
    for key in ("conversation_summary", "memory_summary", "memory_summaries"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            buckets[1].append((key, {"summary": value, "memory_type": key}))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    buckets[1].append((key, item))
    for bucket_index, bucket in enumerate(
        ("episodic_memory_results", "semantic_memory_results"),
        start=2,
    ):
        raw_items = state.get(bucket)
        if raw_items is None:
            raw_items = []
        if not isinstance(raw_items, list):
            raise ContextProviderError(
                provider=MemoryContextProvider.name,
                source_type=MemoryContextProvider.source_type,
                stage="decode_state",
                message=f"{bucket} must be a list",
                original_exception_type="TypeError",
            )
        for item in raw_items:
            if not isinstance(item, dict):
                raise ContextProviderError(
                    provider=MemoryContextProvider.name,
                    source_type=MemoryContextProvider.source_type,
                    stage="decode_state",
                    message=f"{bucket} item must be a dict",
                    original_exception_type="TypeError",
                )
            buckets[bucket_index].append((bucket, item))
    return _round_robin_memory_buckets(buckets, limit=limit)


def _memory_result_to_item(
    result: dict[str, Any],
    *,
    source_bucket: str,
    index: int,
    thread_id: str | None,
    user_id: str | None,
    max_content_chars: int,
) -> ContextItem:
    memory = result.get("memory")
    if isinstance(memory, dict):
        record = memory
    else:
        record = result
    content = _first_text(record, result, keys=("content", "summary", "text"))
    title = _memory_title(record, source_bucket=source_bucket, index=index)
    memory_type = str(result.get("memory_type") or record.get("memory_type") or "")
    priority = _memory_priority(memory_type, source_bucket)
    score = _optional_score(result.get("score"))
    confidence = _optional_score(record.get("confidence"))
    memory_id = (
        record.get("memory_id")
        or record.get("summary_id")
        or result.get("memory_id")
        or result.get("summary_id")
        or f"{source_bucket}:{index}"
    )
    item_thread_id = _first_identifier(
        record.get("thread_id"),
        result.get("thread_id"),
        thread_id,
    )
    item_user_id = _first_identifier(
        record.get("user_id"),
        result.get("user_id"),
        user_id,
    )
    identity_metadata = {
        key: value
        for key, value in (
            ("thread_id", item_thread_id),
            ("user_id", item_user_id),
        )
        if value
    }
    return make_context_item(
        source_type="memory",
        title=title,
        content=content,
        priority=priority,
        scope="session",
        lifetime="long_term",
        compressible=True,
        can_drop=True,
        disclosure_level="summary",
        relevance_score=score,
        confidence=confidence,
        item_id=stable_item_id(
            source_type="memory",
            title=f"{source_bucket}:{memory_id}",
        ),
        metadata={
            "memory_id": str(memory_id),
            "memory_type": memory_type or source_bucket.replace("_memory_results", ""),
            "score": score,
            "created_at": record.get("created_at", ""),
            "match_reason": result.get("match_reason", ""),
            "source_bucket": source_bucket,
            "purpose": ["continuity", "personalization"],
            **identity_metadata,
        },
        max_content_chars=min(
            max_content_chars,
            _MEMORY_CONTENT_CHAR_CAPS[source_bucket],
        ),
    )


def _memory_title(
    record: dict[str, Any],
    *,
    source_bucket: str,
    index: int,
) -> str:
    label = str(
        record.get("memory_type") or source_bucket.replace("_memory_results", "")
    )
    return f"{label or 'memory'}_{index}"


def _memory_priority(memory_type: str, source_bucket: str) -> int:
    if source_bucket in {
        "conversation_summary",
        "memory_summary",
        "memory_summaries",
    }:
        return 80
    if source_bucket == "selected_memory":
        return 75
    normalized = memory_type.strip().lower()
    if normalized == "preference":
        return 70
    if source_bucket == "semantic_memory_results" or normalized == "semantic":
        return 65
    return 60


def _first_text(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> str:
    for values in (primary, secondary):
        for key in keys:
            value = values.get(key)
            if value:
                return str(value)
    return ""


def _optional_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    score = float(value)
    if score < 0.0 or score > 1.0:
        return None
    return score


def _explicit_user_id(state: dict[str, Any]) -> str | None:
    value = state.get("user_id")
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _first_identifier(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _round_robin_memory_buckets(
    buckets: list[list[tuple[str, dict[str, Any]]]],
    *,
    limit: int,
) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    index = 0
    while len(selected) < limit:
        added = False
        for bucket in buckets:
            if index < len(bucket):
                selected.append(bucket[index])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        index += 1
    return selected
