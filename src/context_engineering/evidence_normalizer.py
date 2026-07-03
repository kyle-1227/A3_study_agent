"""Normalize evidence ContextItem relevance before packing/source filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.context_engineering.schema import ContextItem

_PRIMARY_SCORE_KEYS = (
    "relevance_score",
    "score",
    "similarity_score",
    "coverage_score",
    "support_score",
    "grader_score",
    "evidence_score",
)
_CONFIDENCE_KEY = "confidence"
_CONFIDENCE_MEANING_KEYS = (
    "confidence_kind",
    "confidence_type",
    "confidence_scope",
    "confidence_source",
    "confidence_represents",
    "score_type",
)
_CONFIDENCE_RELEVANCE_MEANINGS = {
    "relevance",
    "support",
    "evidence_support",
    "support_confidence",
    "relevance_confidence",
}
_PERCENT_SCALE_VALUES = {"percent", "percentage", "0-100"}


@dataclass(frozen=True)
class EvidenceNormalizationStats:
    """Safe aggregate telemetry for rejected evidence normalization."""

    evidence_rejected_count: int = 0
    evidence_reject_reasons: dict[str, int] = field(default_factory=dict)
    missing_required_relevance_score_count: int = 0
    invalid_relevance_score_count: int = 0

    def as_event_fields(self) -> dict[str, Any]:
        """Return safe primitive fields for trace/SSE payloads."""
        return {
            "evidence_rejected_count": self.evidence_rejected_count,
            "evidence_reject_reasons": dict(self.evidence_reject_reasons),
            "missing_required_relevance_score_count": (
                self.missing_required_relevance_score_count
            ),
            "invalid_relevance_score_count": self.invalid_relevance_score_count,
        }


def normalize_evidence_candidate_score(candidate: dict[str, Any]) -> float | None:
    """Return a normalized 0..1 relevance score for an evidence candidate."""
    score, _reason = _score_from_mapping(candidate)
    return score


def normalize_evidence_item(
    item: ContextItem,
) -> tuple[ContextItem | None, str | None]:
    """Normalize one evidence item, or return a safe reject reason."""
    if item.source_type != "evidence":
        return item, None
    if item.relevance_score is not None:
        return item, None

    metadata = dict(item.metadata or {})
    score, reason = _score_from_mapping(metadata)
    if score is None and _confidence_means_relevance(metadata):
        confidence_score, valid = _normalize_score(
            metadata.get(_CONFIDENCE_KEY),
            metadata,
            _CONFIDENCE_KEY,
        )
        if valid:
            score = confidence_score
            reason = None
        else:
            reason = "invalid_relevance_score"
    if (
        score is None
        and item.confidence is not None
        and _confidence_means_relevance(metadata)
    ):
        confidence_score, valid = _normalize_score(
            item.confidence,
            metadata,
            _CONFIDENCE_KEY,
        )
        if valid:
            score = confidence_score
            reason = None
        else:
            reason = "invalid_relevance_score"
    if score is None:
        return None, reason or "missing_required_relevance_score"

    metadata["relevance_score"] = score
    data = item.model_dump()
    data["relevance_score"] = score
    data["metadata"] = metadata
    return ContextItem.model_validate(data), None


def normalize_evidence_items(
    items: list[ContextItem],
) -> tuple[list[ContextItem], EvidenceNormalizationStats]:
    """Normalize all evidence items and collect safe rejection telemetry."""
    normalized: list[ContextItem] = []
    reject_reasons: dict[str, int] = {}
    missing_count = 0
    invalid_count = 0
    for item in items:
        normalized_item, reason = normalize_evidence_item(item)
        if normalized_item is not None:
            normalized.append(normalized_item)
            continue
        reason = reason or "missing_required_relevance_score"
        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
        if reason == "missing_required_relevance_score":
            missing_count += 1
        elif reason == "invalid_relevance_score":
            invalid_count += 1
    return normalized, EvidenceNormalizationStats(
        evidence_rejected_count=sum(reject_reasons.values()),
        evidence_reject_reasons=reject_reasons,
        missing_required_relevance_score_count=missing_count,
        invalid_relevance_score_count=invalid_count,
    )


def _score_from_mapping(mapping: dict[str, Any]) -> tuple[float | None, str | None]:
    found_invalid = False
    for key in _PRIMARY_SCORE_KEYS:
        if key not in mapping:
            continue
        score, valid = _normalize_score(mapping.get(key), mapping, key)
        if valid:
            return score, None
        found_invalid = True
    if _CONFIDENCE_KEY in mapping and _confidence_means_relevance(mapping):
        score, valid = _normalize_score(
            mapping.get(_CONFIDENCE_KEY),
            mapping,
            _CONFIDENCE_KEY,
        )
        if valid:
            return score, None
        found_invalid = True
    if found_invalid:
        return None, "invalid_relevance_score"
    return None, "missing_required_relevance_score"


def _normalize_score(
    value: Any,
    mapping: dict[str, Any],
    key: str,
) -> tuple[float | None, bool]:
    raw = _coerce_number(value)
    if raw is None or raw < 0:
        return None, False
    if raw <= 1.0:
        return raw, True
    scale = _explicit_scale(mapping, key)
    if scale is None or scale <= 1.0:
        return None, False
    if raw > scale:
        return None, False
    return raw / scale, True


def _explicit_scale(mapping: dict[str, Any], key: str) -> float | None:
    for scale_key in (f"{key}_max", "score_max"):
        scale = _coerce_number(mapping.get(scale_key))
        if scale is not None and scale > 1.0:
            return scale
    for scale_key in (f"{key}_scale", "score_scale"):
        scale_value = mapping.get(scale_key)
        scale = _coerce_number(scale_value)
        if scale is not None and scale > 1.0:
            return scale
        text = str(scale_value or "").strip().lower()
        if text in _PERCENT_SCALE_VALUES:
            return 100.0
    for percent_key in (f"{key}_is_percent", "score_is_percent", "is_percent_score"):
        if mapping.get(percent_key) is True:
            return 100.0
    return None


def _confidence_means_relevance(mapping: dict[str, Any]) -> bool:
    if mapping.get("confidence_is_relevance") is True:
        return True
    for key in _CONFIDENCE_MEANING_KEYS:
        text = str(mapping.get(key) or "").strip().lower()
        if text in _CONFIDENCE_RELEVANCE_MEANINGS:
            return True
    return False


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None
