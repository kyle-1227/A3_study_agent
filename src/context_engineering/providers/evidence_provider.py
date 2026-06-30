"""Context provider for already-collected evidence candidates."""

from __future__ import annotations

from typing import Any

from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.schema import ContextItem, ContextProviderError


class EvidenceContextProvider:
    """Objectize local/web evidence already present in graph state."""

    name = "evidence_provider"
    source_type = "evidence"

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        if context.max_items_per_provider <= 0:
            return []
        try:
            candidates = _existing_evidence_candidates(
                context.state,
                limit=context.max_items_per_provider,
            )
            return [
                _candidate_to_item(
                    candidate,
                    index=index,
                    max_content_chars=context.max_content_chars_per_item,
                )
                for index, candidate in enumerate(candidates)
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


def _existing_evidence_candidates(
    state: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for bucket in (
        "evidence_candidates",
        "local_evidence_candidates",
        "web_evidence_candidates",
    ):
        raw_items = state.get(bucket) or []
        if not isinstance(raw_items, list):
            raise ContextProviderError(
                provider=EvidenceContextProvider.name,
                source_type=EvidenceContextProvider.source_type,
                stage="decode_state",
                message=f"{bucket} must be a list",
                original_exception_type="TypeError",
            )
        for item in raw_items:
            if not isinstance(item, dict):
                raise ContextProviderError(
                    provider=EvidenceContextProvider.name,
                    source_type=EvidenceContextProvider.source_type,
                    stage="decode_state",
                    message=f"{bucket} item must be a dict",
                    original_exception_type="TypeError",
                )
            evidence_id = str(item.get("evidence_id") or item.get("source_id") or "")
            dedupe_key = evidence_id or f"{bucket}:{len(candidates)}"
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            candidates.append({**item, "_context_source_bucket": bucket})
            if len(candidates) >= limit:
                return candidates
    return candidates


def _candidate_to_item(
    candidate: dict[str, Any],
    *,
    index: int,
    max_content_chars: int,
) -> ContextItem:
    score = _candidate_score(candidate)
    source_type = str(candidate.get("source_type") or candidate.get("type") or "")
    evidence_id = str(
        candidate.get("evidence_id") or candidate.get("source_id") or index
    )
    title = str(
        candidate.get("title") or candidate.get("source") or f"evidence_{index}"
    )
    content = _candidate_content(candidate)
    return make_context_item(
        source_type="evidence",
        title=title,
        content=content,
        priority=_evidence_priority(score),
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        relevance_score=score,
        metadata={
            "evidence_id": evidence_id,
            "source_type": source_type,
            "provider": candidate.get("provider", ""),
            "url": candidate.get("url", ""),
            "title": title,
            "rank": index,
            "retrieval_mode": candidate.get("_context_source_bucket", ""),
            "score": score,
        },
        max_content_chars=max_content_chars,
    )


def _candidate_score(candidate: dict[str, Any]) -> float | None:
    for key in ("score", "rerank_score", "raw_vector_score"):
        score = _safe_score(candidate.get(key))
        if score is not None:
            return score
    return None


def _candidate_content(candidate: dict[str, Any]) -> str:
    for key in (
        "content_preview",
        "snippet",
        "excerpt",
        "summary",
        "content",
        "text",
        "page_content",
        "coverage_contribution",
    ):
        value = candidate.get(key)
        if value:
            return str(value)
    return ""


def _safe_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    score = float(value)
    if score < 0.0 or score > 1.0:
        return None
    return score


def _evidence_priority(score: float | None) -> int:
    if score is None:
        return 75
    if score >= 0.7:
        return 85
    if score >= 0.4:
        return 75
    return 65
