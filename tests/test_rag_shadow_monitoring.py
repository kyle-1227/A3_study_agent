from __future__ import annotations

from pydantic import ValidationError
import pytest

from src.rag.monitoring import (
    GenerationValidationFingerprint,
    RetrievalHealthEvent,
    assess_validation_reuse,
)
from src.rag.shadow import ShadowComparable, execute_shadow


def _summary(value: dict[str, int]) -> ShadowComparable:
    return ShadowComparable(
        status="ok",
        evidence_count=value["evidence"],
        parent_count=value["parents"],
        context_tokens=value["tokens"],
    )


def test_shadow_candidate_failure_is_visible_and_primary_output_is_served() -> None:
    primary = {"answer": "served", "evidence": 2, "parents": 1, "tokens": 50}

    def candidate() -> dict[str, int]:
        raise RuntimeError("provider response body must not be recorded")

    result = execute_shadow(
        request_id="request-a",
        subject="math",
        primary_generation_id="gen-primary",
        candidate_generation_id="gen-candidate",
        primary_call=lambda: primary,
        candidate_call=candidate,
        primary_summarizer=_summary,
        candidate_summarizer=_summary,
    )

    assert result.served_output is primary
    assert result.record.candidate_status == "failed"
    assert result.record.candidate_failure_type == "RuntimeError"
    assert "provider response body" not in result.record.model_dump_json()


def test_shadow_comparison_contains_only_metric_deltas() -> None:
    result = execute_shadow(
        request_id="request-b",
        subject="python",
        primary_generation_id="gen-primary",
        candidate_generation_id="gen-candidate",
        primary_call=lambda: {"evidence": 2, "parents": 1, "tokens": 50},
        candidate_call=lambda: {"evidence": 3, "parents": 2, "tokens": 70},
        primary_summarizer=_summary,
        candidate_summarizer=_summary,
    )
    assert result.record.evidence_count_delta == 1
    assert result.record.parent_count_delta == 1
    assert result.record.context_token_delta == 20


def _fingerprint(character: str) -> GenerationValidationFingerprint:
    return GenerationValidationFingerprint(
        schema_version="generation_validation_fingerprint_v1",
        source_fingerprint=character * 64,
        subject_fingerprint="b" * 64,
        policy_fingerprint="c" * 64,
        embedding_fingerprint="d" * 64,
        benchmark_dataset_fingerprint="e" * 64,
    )


def test_validation_reuse_is_invalidated_by_any_output_identity_change() -> None:
    validated = _fingerprint("a")
    assert assess_validation_reuse(validated=validated, current=validated).reusable

    changed = _fingerprint("f")
    decision = assess_validation_reuse(validated=validated, current=changed)
    assert decision.reusable is False
    assert decision.invalidation_codes == ("source_changed",)


def test_health_event_forbids_raw_fields_and_requires_failure_reason() -> None:
    payload = {
        "schema_version": "retrieval_health_event_v1",
        "request_id_hash": "a" * 64,
        "generation_id": "gen-a",
        "retrieval_fingerprint": "b" * 64,
        "subject": "math",
        "route_kind": "candidate",
        "status": "failed",
        "failure_reason_code": "reranker_protocol_error",
        "vector_ms": 1.0,
        "bm25_ms": 1.0,
        "reranker_ms": 1.0,
        "aggregate_ms": 0.1,
        "hydrate_ms": 0.0,
        "total_ms": 3.1,
        "child_hit_count": 0,
        "parent_hit_count": 0,
        "context_tokens": 0,
        "judge_keep_count": 0,
        "orphan_child_count": 0,
        "generation_mismatch_count": 0,
        "parent_hydration_failure_count": 0,
    }
    event = RetrievalHealthEvent.model_validate(payload)
    assert "content" not in event.model_dump()

    with pytest.raises(ValidationError):
        RetrievalHealthEvent.model_validate({**payload, "raw_body": "secret"})
    with pytest.raises(ValidationError):
        RetrievalHealthEvent.model_validate({**payload, "failure_reason_code": None})
