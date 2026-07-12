"""Content-free shadow and health contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rag.evidence_observability import (
    EvidenceOrchestrationHealthEvent,
    EvidenceShadowRecord,
    EvidenceShadowSummary,
    request_id_hash,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def _summary() -> EvidenceShadowSummary:
    return EvidenceShadowSummary(
        schema_version="evidence_shadow_summary_v1",
        terminal_status="partial_resources_ready",
        requirement_count=4,
        complete_count=2,
        partial_count=1,
        missing_count=1,
        round_count=2,
        search_task_count=6,
        ready_resource_count=1,
        blocked_resource_count=1,
        ledger_count=8,
    )


def test_shadow_and_health_contracts_are_content_free():
    summary = _summary()
    record = EvidenceShadowRecord(
        schema_version="evidence_shadow_record_v1",
        request_id_hash=request_id_hash("request-1"),
        primary_graph_fingerprint=HASH_A,
        candidate_bundle_fingerprint=HASH_B,
        primary_summary=summary,
        candidate_status="ok",
        candidate_summary=summary,
        candidate_failure_type=None,
        primary_latency_ms=100.0,
        candidate_latency_ms=120.0,
    )
    health = EvidenceOrchestrationHealthEvent(
        schema_version="evidence_orchestration_health_v1",
        request_id_hash=record.request_id_hash,
        generation_id="generation-a",
        candidate_bundle_fingerprint=HASH_B,
        route_kind="shadow",
        status="ok",
        terminal_status="partial_resources_ready",
        failure_reason_code=None,
        round_count=2,
        search_task_count=6,
        ready_resource_count=1,
        blocked_resource_count=1,
        ledger_count=8,
        total_latency_ms=120.0,
        candidate_failure_policy="fail_fast",
    )

    assert record.candidate_summary == summary
    assert health.candidate_failure_policy == "fail_fast"


def test_shadow_contract_rejects_query_url_and_provider_body_fields():
    payload = {
        **_summary().model_dump(mode="python"),
        "query": "private query",
        "url": "https://example.invalid",
        "provider_body": "private body",
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidenceShadowSummary.model_validate(payload)


def test_failed_health_event_cannot_carry_a_success_terminal_status():
    with pytest.raises(ValidationError, match="failure reason code"):
        EvidenceOrchestrationHealthEvent(
            schema_version="evidence_orchestration_health_v1",
            request_id_hash=request_id_hash("request-2"),
            generation_id="generation-a",
            candidate_bundle_fingerprint=HASH_B,
            route_kind="candidate",
            status="failed",
            terminal_status="sufficient",
            failure_reason_code=None,
            round_count=1,
            search_task_count=2,
            ready_resource_count=0,
            blocked_resource_count=1,
            ledger_count=0,
            total_latency_ms=20.0,
            candidate_failure_policy="fail_fast",
        )
