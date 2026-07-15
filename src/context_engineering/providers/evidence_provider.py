"""Context provider for already-collected evidence candidates."""

from __future__ import annotations

from typing import Any

from src.config.evidence_orchestration_contracts import ResourceEvidenceAssignment
from src.context_engineering.evidence_normalizer import (
    normalize_evidence_candidate_score,
)
from src.context_engineering.itemizer import make_context_item
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.schema import ContextItem, ContextProviderError

_SCORE_METADATA_KEYS = (
    "relevance_score",
    "score",
    "similarity_score",
    "coverage_score",
    "support_score",
    "grader_score",
    "evidence_score",
    "confidence",
    "confidence_kind",
    "confidence_type",
    "confidence_scope",
    "confidence_source",
    "confidence_represents",
    "confidence_is_relevance",
    "score_source",
    "score_type",
    "score_reason",
)

_ASSIGNMENT_APPROVED_KEY = "_resource_assignment_approved"


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
    assigned_evidence_ids = _assigned_evidence_ids(state)
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for bucket in (
        "graded_evidence",
        "evidence_items",
        "task_workspace.evidence_summaries",
        "local_evidence",
        "web_evidence",
        "retrieval_evidence",
        "evidence_candidates",
        "local_evidence_candidates",
        "web_evidence_candidates",
    ):
        raw_items = (
            _workspace_evidence_summaries(state)
            if bucket == "task_workspace.evidence_summaries"
            else state.get(bucket) or []
        )
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
            if (
                assigned_evidence_ids is not None
                and evidence_id not in assigned_evidence_ids
            ):
                continue
            dedupe_key = evidence_id or f"{bucket}:{len(candidates)}"
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            candidate = {**item, "_context_source_bucket": bucket}
            if assigned_evidence_ids is not None:
                candidate[_ASSIGNMENT_APPROVED_KEY] = True
            candidates.append(candidate)
            if len(candidates) >= limit:
                return candidates
    if assigned_evidence_ids is not None and not candidates:
        raise ContextProviderError(
            provider=EvidenceContextProvider.name,
            source_type=EvidenceContextProvider.source_type,
            stage="assignment_filter",
            message="resource evidence assignment resolved to no provider candidates",
            original_exception_type="EvidenceAssignmentResolutionError",
        )
    return candidates


def _assigned_evidence_ids(state: dict[str, Any]) -> frozenset[str] | None:
    if "resource_evidence_assignment" not in state:
        return None
    assignment = ResourceEvidenceAssignment.model_validate(
        state["resource_evidence_assignment"]
    )
    return frozenset(assignment.evidence_ids)


def _workspace_evidence_summaries(state: dict[str, Any]) -> list[dict[str, Any]]:
    workspace = state.get("task_workspace")
    if not isinstance(workspace, dict):
        return []
    raw_items = workspace.get("evidence_summaries") or []
    if not isinstance(raw_items, list):
        return []
    result: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        if item.get("purpose") != "factual_grounding":
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not evidence_id or not summary:
            continue
        result.append(item)
    return result


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
    workspace_item = (
        candidate.get("_context_source_bucket") == "task_workspace.evidence_summaries"
    )
    metadata = {
        "evidence_id": evidence_id,
        "source_type": source_type,
        "provider": candidate.get("provider", ""),
        "url": candidate.get("url", ""),
        "title": title,
        "rank": index,
        "retrieval_mode": candidate.get("_context_source_bucket", ""),
        "purpose": candidate.get("purpose", ""),
        "subject": candidate.get("subject", ""),
        "normalized_subject": candidate.get("normalized_subject", ""),
        "thread_id": candidate.get("thread_id", ""),
        "request_id": candidate.get("request_id", ""),
        "created_at": candidate.get("created_at", ""),
        "grounding_approved": bool(
            workspace_item
            or candidate.get(_ASSIGNMENT_APPROVED_KEY) is True
            or candidate.get("judge_keep") is True
            or candidate.get("keep") is True
        ),
    }
    metadata.update(_candidate_score_metadata(candidate))
    if score is not None:
        metadata["score"] = score
    return make_context_item(
        source_type="evidence",
        title=title,
        content=content,
        priority=_evidence_priority(score, workspace=workspace_item),
        scope="session" if workspace_item else "turn",
        lifetime="session" if workspace_item else "turn",
        compressible=True,
        can_drop=True,
        disclosure_level="summary" if workspace_item else "snippet",
        relevance_score=_workspace_relevance(score) if workspace_item else score,
        metadata=metadata,
        max_content_chars=max_content_chars,
    )


def _candidate_score(candidate: dict[str, Any]) -> float | None:
    return normalize_evidence_candidate_score(candidate)


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


def _candidate_score_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in _SCORE_METADATA_KEYS:
        if key in candidate:
            metadata[key] = candidate[key]
        max_key = f"{key}_max"
        if max_key in candidate:
            metadata[max_key] = candidate[max_key]
        scale_key = f"{key}_scale"
        if scale_key in candidate:
            metadata[scale_key] = candidate[scale_key]
        percent_key = f"{key}_is_percent"
        if percent_key in candidate:
            metadata[percent_key] = candidate[percent_key]
    for key in ("score_max", "score_scale", "score_is_percent", "is_percent_score"):
        if key in candidate:
            metadata[key] = candidate[key]
    return metadata


def _evidence_priority(score: float | None, *, workspace: bool = False) -> int:
    if workspace:
        if score is None:
            return 58
        if score >= 0.7:
            return 66
        if score >= 0.4:
            return 60
        return 55
    if score is None:
        return 75
    if score >= 0.7:
        return 85
    if score >= 0.4:
        return 75
    return 65


def _workspace_relevance(score: float | None) -> float:
    if score is None:
        return 0.45
    return max(0.35, min(float(score) * 0.85, 0.75))
