"""Shadow-only context importance scoring contracts and aggregation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.context_engineering.packing.apply import (
    ImportanceScoringPolicy,
    sanitize_context_content,
)
from src.context_engineering.schema import ContextItem, sanitize_error_message

ContextImportanceReasonCode = Literal[
    "directly_supports_resource",
    "matches_user_goal",
    "required_constraint",
    "high_quality_evidence",
    "useful_memory",
    "low_relevance",
    "duplicate_context",
    "too_generic",
    "unsafe_or_instruction_like",
    "over_budget_candidate",
    "unknown",
]
_ALLOWED_REASON_CODES = (
    "directly_supports_resource",
    "matches_user_goal",
    "required_constraint",
    "high_quality_evidence",
    "useful_memory",
    "low_relevance",
    "duplicate_context",
    "too_generic",
    "unsafe_or_instruction_like",
    "over_budget_candidate",
    "unknown",
)


class ContextImportanceError(RuntimeError):
    """Shadow importance scorer failure with sanitized diagnostics."""

    def __init__(self, *, reason: str, warning: object, error_type: str = "") -> None:
        self.reason = str(reason or "").strip() or "context_importance_failed"
        self.warning = sanitize_error_message(warning)
        self.error_type = str(error_type or "").strip()
        super().__init__(f"{self.reason}: {self.warning}")


class ContextImportanceScore(BaseModel):
    """Strict LLM-produced score for one candidate item."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    reason_code: ContextImportanceReasonCode


class ContextImportanceScores(BaseModel):
    """Strict response schema for the raw importance scorer."""

    model_config = ConfigDict(extra="forbid")

    scores: list[ContextImportanceScore]


@dataclass(frozen=True)
class ContextImportanceTelemetry:
    """Safe aggregate telemetry for context_importance_scored."""

    source_counts: dict[str, int]
    score_buckets: dict[str, int]
    reason_code_counts: dict[str, int]
    candidate_count: int
    scored_count: int
    kept_count: int
    dropped_count: int
    scoring_elapsed_ms: float
    disabled_reason: str = ""
    error_reason: str = ""
    error_type: str = ""
    warnings: list[str] = field(default_factory=list)


def build_importance_scorer_messages(
    *,
    items: list[ContextItem],
    policy: ImportanceScoringPolicy,
) -> list[dict[str, str]]:
    """Build internal scorer messages; callers must never trace these."""
    scorer_items = []
    for item in items[: policy.max_items_to_score]:
        scorer_items.append(
            {
                "item_id": item.id,
                "source_type": item.source_type,
                "sanitized_title": sanitize_error_message(
                    item.title or item.id,
                    max_chars=120,
                ),
                "token_estimate": item.token_estimate,
                "priority": item.priority,
                "relevance_score": item.relevance_score,
                "confidence": item.confidence,
                "recency_score": item.recency_score,
                "disclosure_level": item.disclosure_level,
                "content_preview": sanitize_context_content(
                    item.content,
                    max_chars=policy.max_content_preview_chars,
                ),
            }
        )
    payload = json.dumps({"items": scorer_items}, ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "Score each context item for usefulness to the next answer. "
                'Return strict JSON: {"scores":[{"item_id":"...",'
                '"score":0.0,"reason_code":"..."}]}. '
                "Allowed reason_code values: " + ", ".join(_ALLOWED_REASON_CODES) + ". "
                "Do not add extra keys."
            ),
        },
        {"role": "user", "content": payload},
    ]


def parse_importance_scorer_output(raw_output: object) -> ContextImportanceScores:
    """Strictly parse LLM-produced JSON scorer output."""
    raw = str(raw_output or "").strip()
    if not raw:
        raise ContextImportanceError(
            reason="context_importance_json_parse_failed",
            warning="importance scorer returned empty output",
            error_type="ValueError",
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContextImportanceError(
            reason="context_importance_json_parse_failed",
            warning="importance scorer output was not valid JSON",
            error_type=type(exc).__name__,
        ) from exc
    try:
        return ContextImportanceScores.model_validate(data)
    except ValidationError as exc:
        raise ContextImportanceError(
            reason="context_importance_schema_invalid",
            warning="importance scorer output did not match schema",
            error_type=type(exc).__name__,
        ) from exc


def aggregate_importance_success(
    *,
    items: list[ContextItem],
    scores: ContextImportanceScores,
    policy: ImportanceScoringPolicy,
    started_at: float,
) -> ContextImportanceTelemetry:
    """Aggregate scorer output without exposing per-item details."""
    item_ids = {item.id for item in items[: policy.max_items_to_score]}
    scored = [score for score in scores.scores if score.item_id in item_ids]
    kept_count = sum(
        1 for score in scored if score.score >= policy.min_shadow_score_for_analysis
    )
    return ContextImportanceTelemetry(
        source_counts=_source_counts(items),
        score_buckets=_score_buckets(scored),
        reason_code_counts=_reason_code_counts(scored),
        candidate_count=len(items),
        scored_count=len(scored),
        kept_count=kept_count,
        dropped_count=max(len(scored) - kept_count, 0),
        scoring_elapsed_ms=_elapsed_ms(started_at),
    )


def aggregate_importance_failure(
    *,
    items: list[ContextItem],
    started_at: float | None,
    reason: str,
    error_type: str = "",
    warning: object = "",
) -> ContextImportanceTelemetry:
    """Build safe aggregate telemetry for disabled or failed scoring."""
    return ContextImportanceTelemetry(
        source_counts=_source_counts(items),
        score_buckets={},
        reason_code_counts={reason: 1} if reason else {},
        candidate_count=len(items),
        scored_count=0,
        kept_count=0,
        dropped_count=0,
        scoring_elapsed_ms=_elapsed_ms(started_at) if started_at is not None else 0.0,
        disabled_reason=reason
        if reason.startswith("context_importance_scorer")
        else "",
        error_reason=reason
        if not reason.startswith("context_importance_scorer")
        else "",
        error_type=error_type,
        warnings=[sanitize_error_message(warning)] if warning else [],
    )


def _source_counts(items: list[ContextItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        source = str(item.source_type)
        counts[source] = counts.get(source, 0) + 1
    return counts


def _score_buckets(scores: list[ContextImportanceScore]) -> dict[str, int]:
    buckets = {
        "0.00-0.25": 0,
        "0.25-0.50": 0,
        "0.50-0.75": 0,
        "0.75-1.00": 0,
    }
    for score in scores:
        if score.score < 0.25:
            buckets["0.00-0.25"] += 1
        elif score.score < 0.5:
            buckets["0.25-0.50"] += 1
        elif score.score < 0.75:
            buckets["0.50-0.75"] += 1
        else:
            buckets["0.75-1.00"] += 1
    return buckets


def _reason_code_counts(scores: list[ContextImportanceScore]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for score in scores:
        reason_code = sanitize_error_message(score.reason_code, max_chars=80)
        counts[reason_code] = counts.get(reason_code, 0) + 1
    return counts


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
