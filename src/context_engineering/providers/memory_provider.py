"""Context provider for already-retrieved memory state."""

from __future__ import annotations

from typing import Any

from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.schema import ContextItem, ContextProviderError


class MemoryContextProvider:
    """Objectize memory results already present in graph state."""

    name = "memory_provider"
    source_type = "memory"

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        if context.max_items_per_provider <= 0:
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
    results: list[tuple[str, dict[str, Any]]] = []
    selected_memory = state.get("selected_memory")
    if isinstance(selected_memory, dict):
        results.append(("selected_memory", selected_memory))
    elif isinstance(selected_memory, list):
        for item in selected_memory:
            if isinstance(item, dict):
                results.append(("selected_memory", item))
                if len(results) >= limit:
                    return results
    for key in ("conversation_summary", "memory_summary", "memory_summaries"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            results.append((key, {"summary": value, "memory_type": key}))
            if len(results) >= limit:
                return results
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    results.append((key, item))
                    if len(results) >= limit:
                        return results
    for bucket in ("episodic_memory_results", "semantic_memory_results"):
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
            results.append((bucket, item))
            if len(results) >= limit:
                return results
    return results


def _memory_result_to_item(
    result: dict[str, Any],
    *,
    source_bucket: str,
    index: int,
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
        metadata={
            "memory_id": str(memory_id),
            "memory_type": memory_type or source_bucket.replace("_memory_results", ""),
            "score": score,
            "created_at": record.get("created_at", ""),
            "match_reason": result.get("match_reason", ""),
            "source_bucket": source_bucket,
        },
        max_content_chars=max_content_chars,
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
