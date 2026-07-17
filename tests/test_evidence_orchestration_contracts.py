"""Strict configuration and business-contract tests for evidence orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config._rag_config import RagConfigValidationError
from src.config.evidence_orchestration_config import (
    EvidenceSourcePolicy,
    load_evidence_orchestration_config,
    load_resource_evidence_profiles,
)
from src.config.evidence_orchestration_contracts import (
    DuplicateRetrievalSignatureError,
    EvidenceLedgerEntry,
    EvidenceRequirementValidationError,
    EvidenceRequirementDraft,
    EvidenceRequirementDraftBatch,
    RequirementCoverage,
    RequirementCoverageBatch,
    RequirementCoverageValidationError,
    RetrievalTaskValidationError,
    build_retrieval_task,
    compile_evidence_requirement_batch,
    compile_requirement_coverage_batch,
    derive_resource_evidence_assignments,
    derive_resource_readiness,
    make_evidence_id,
    validate_evidence_ledger,
    validate_requirement_coverage,
    validate_requirement_inventory,
    validate_retrieval_tasks,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config" / "rag" / "evidence_orchestration.yaml"
PROFILES_PATH = ROOT / "config" / "rag" / "resource_evidence_profiles.yaml"


def _quiz_requirements():
    profiles = load_resource_evidence_profiles(PROFILES_PATH)
    profile = profiles.profile_for("quiz")
    batch = EvidenceRequirementDraftBatch(
        schema_version="evidence_requirement_draft_batch_v1",
        requirements=[
            EvidenceRequirementDraft(
                resource_type="quiz",
                subject="math",
                topic_id="math.algebra",
                profile_need_id=need.need_id,
                evidence_kind=need.evidence_kind,
                scope=need.scope,
                criticality=need.criticality,
                source_policy=need.source_policy,
                acceptance_criteria=need.acceptance_criteria,
                query_intent=f"math {need.evidence_kind} retrieval",
            )
            for need in profile.needs
        ],
    )
    return profiles, compile_evidence_requirement_batch(batch)


def _requirements_for_source_policies(
    *source_policies: EvidenceSourcePolicy,
):
    return compile_evidence_requirement_batch(
        EvidenceRequirementDraftBatch(
            schema_version="evidence_requirement_draft_batch_v1",
            requirements=[
                EvidenceRequirementDraft(
                    resource_type="quiz",
                    subject="math",
                    topic_id=f"math.policy_{index}",
                    profile_need_id=f"policy_need_{index}",
                    evidence_kind=f"policy_evidence_{index}",
                    scope="per_subject",
                    criticality="required",
                    source_policy=source_policy,
                    acceptance_criteria=(
                        f"Supports the explicit source policy for requirement {index}."
                    ),
                    query_intent=f"math policy evidence {index}",
                )
                for index, source_policy in enumerate(source_policies)
            ],
        )
    )


def test_strict_configs_load_complete_explicit_inventory():
    policy = load_evidence_orchestration_config(POLICY_PATH)
    profiles = load_resource_evidence_profiles(PROFILES_PATH)

    assert policy.max_supplement_rounds == 2
    assert policy.max_search_tasks_per_round == 6
    assert policy.max_total_search_tasks == 18
    assert policy.max_concurrent_tasks == 1
    assert policy.web_timeout_seconds == 120.0
    assert policy.required_task_priority == "high"
    assert policy.supporting_task_priority == "medium"
    assert policy.retrieval_priority_weights.weight_for("high") == 1.0
    assert policy.retrieval_priority_weights.weight_for("medium") == 0.7
    assert policy.retrieval_priority_weights.weight_for("low") == 0.4
    assert policy.candidate_failure_policy == "fail_fast"
    assert policy.source_error_policy == "fail_fast"
    assert tuple(profile.resource_type for profile in profiles.profiles) == (
        "review_doc",
        "mindmap",
        "quiz",
        "code_practice",
        "video_script",
        "video_animation",
        "study_plan",
    )
    assert all(profile.needs for profile in profiles.profiles)


def test_strict_config_rejects_extra_field_without_defaulting():
    invalid = Path("invalid-evidence-orchestration.yaml")
    invalid_text = POLICY_PATH.read_text(encoding="utf-8") + "unexpected_field: true\n"

    with (
        patch.object(Path, "read_text", return_value=invalid_text),
        pytest.raises(RagConfigValidationError) as exc_info,
    ):
        load_evidence_orchestration_config(invalid)

    assert any(
        location == "unexpected_field"
        for location, _error_type in exc_info.value.validation_errors
    )


def test_strict_config_rejects_non_descending_priority_weights():
    invalid = Path("invalid-evidence-priority-weights.yaml")
    invalid_text = POLICY_PATH.read_text(encoding="utf-8").replace(
        "  medium: 0.7",
        "  medium: 1.0",
    )

    with (
        patch.object(Path, "read_text", return_value=invalid_text),
        pytest.raises(RagConfigValidationError) as exc_info,
    ):
        load_evidence_orchestration_config(invalid)

    assert any(
        location == "retrieval_priority_weights"
        for location, _error_type in exc_info.value.validation_errors
    )


def test_provider_batches_reject_tuple_coercion() -> None:
    profiles = load_resource_evidence_profiles(PROFILES_PATH)
    need = profiles.profile_for("quiz").needs[0]
    draft = EvidenceRequirementDraft(
        resource_type="quiz",
        subject="math",
        topic_id="math.algebra",
        profile_need_id=need.need_id,
        evidence_kind=need.evidence_kind,
        scope=need.scope,
        criticality=need.criticality,
        source_policy=need.source_policy,
        acceptance_criteria=need.acceptance_criteria,
        query_intent="math algebra assessable evidence",
    )
    with pytest.raises(ValidationError, match="list_type"):
        EvidenceRequirementDraftBatch(
            schema_version="evidence_requirement_draft_batch_v1",
            requirements=(draft,),
        )

    coverage = RequirementCoverage(
        requirement_id="requirement-test",
        resource_type="quiz",
        subject="math",
        round_index=0,
        coverage_state="missing",
        evidence_ids=[],
        confidence=0.0,
        reason="No evidence is available.",
        suggested_local_query="math algebra course notes",
        suggested_web_query="",
    )
    with pytest.raises(ValidationError, match="list_type"):
        RequirementCoverageBatch(
            schema_version="requirement_coverage_batch_v1",
            round_index=0,
            coverages=(coverage,),
        )

    coverage_payload = coverage.model_dump(mode="python")
    coverage_payload["evidence_ids"] = ()
    with pytest.raises(ValidationError, match="list_type"):
        RequirementCoverage.model_validate(coverage_payload)

    compiled = compile_requirement_coverage_batch(
        RequirementCoverageBatch(
            schema_version="requirement_coverage_batch_v1",
            round_index=0,
            coverages=[coverage],
        )
    )
    assert isinstance(compiled.coverages, tuple)
    assert isinstance(compiled.coverages[0].evidence_ids, tuple)
    with pytest.raises(ValidationError, match="frozen_instance"):
        setattr(compiled.coverages[0], "evidence_ids", ())


def test_requirement_inventory_exactly_covers_profile_slots():
    policy = load_evidence_orchestration_config(POLICY_PATH)
    profiles, requirements = _quiz_requirements()

    validate_requirement_inventory(
        requested_resource_types=("quiz",),
        requested_subjects=("math",),
        canonical_subjects={"math"},
        requirements=requirements,
        profiles=profiles,
        config=policy,
    )

    with pytest.raises(
        EvidenceRequirementValidationError,
        match="requirement_inventory_mismatch",
    ) as exc_info:
        validate_requirement_inventory(
            requested_resource_types=("quiz",),
            requested_subjects=("math",),
            canonical_subjects={"math"},
            requirements=requirements[:1],
            profiles=profiles,
            config=policy,
        )

    missing = requirements[1]
    message = str(exc_info.value)
    assert (
        "missing_slots=[resource_type=quiz|subject=math|"
        f"profile_need_id={missing.profile_need_id}]"
    ) in message
    assert "unexpected_slots=[]" in message


def test_requirement_inventory_reports_missing_and_unexpected_slots():
    policy = load_evidence_orchestration_config(POLICY_PATH)
    profiles, requirements = _quiz_requirements()
    replaced = requirements[1]
    unexpected = compile_evidence_requirement_batch(
        EvidenceRequirementDraftBatch(
            schema_version="evidence_requirement_draft_batch_v1",
            requirements=[
                EvidenceRequirementDraft(
                    resource_type=replaced.resource_type,
                    subject="physics",
                    topic_id="physics.mechanics",
                    profile_need_id=replaced.profile_need_id,
                    evidence_kind=replaced.evidence_kind,
                    scope=replaced.scope,
                    criticality=replaced.criticality,
                    source_policy=replaced.source_policy,
                    acceptance_criteria=replaced.acceptance_criteria,
                    query_intent="physics mechanics profile evidence",
                )
            ],
        )
    )[0]

    with pytest.raises(EvidenceRequirementValidationError) as unexpected_exc:
        validate_requirement_inventory(
            requested_resource_types=("quiz",),
            requested_subjects=("math",),
            canonical_subjects={"math", "physics"},
            requirements=(requirements[0], unexpected),
            profiles=profiles,
            config=policy,
        )

    message = str(unexpected_exc.value)
    expected_slot = f"profile_need_id={replaced.profile_need_id}"
    assert f"missing_slots=[resource_type=quiz|subject=math|{expected_slot}]" in message
    assert (
        f"unexpected_slots=[resource_type=quiz|subject=physics|{expected_slot}]"
        in message
    )


def test_retrieval_task_validation_rejects_illegal_source_and_repeat():
    policy = load_evidence_orchestration_config(POLICY_PATH)
    _profiles, requirements = _quiz_requirements()
    local_only = next(
        item for item in requirements if item.source_policy == "local_only"
    )
    illegal = build_retrieval_task(
        requirement=local_only,
        source_type="web",
        query="math assessable facts official source",
        purpose=local_only.acceptance_criteria,
        priority="high",
        round_index=0,
        result_limit=policy.max_results_per_task,
    )

    with pytest.raises(
        RetrievalTaskValidationError,
        match="illegal_source_for_requirement",
    ):
        validate_retrieval_tasks(
            tasks=(illegal,),
            requirements=requirements,
            config=policy,
            round_index=0,
            existing_total_search_tasks=0,
            prior_retrieval_signatures=set(),
            local_then_web_gap_requirement_ids=set(),
        )

    legal = build_retrieval_task(
        requirement=local_only,
        source_type="local_rag",
        query="math assessable facts course notes",
        purpose=local_only.acceptance_criteria,
        priority="high",
        round_index=0,
        result_limit=policy.max_results_per_task,
    )
    with pytest.raises(
        DuplicateRetrievalSignatureError,
        match="duplicate_retrieval_signature",
    ):
        validate_retrieval_tasks(
            tasks=(legal, legal),
            requirements=requirements,
            config=policy,
            round_index=0,
            existing_total_search_tasks=0,
            prior_retrieval_signatures=set(),
            local_then_web_gap_requirement_ids=set(),
        )


def test_coverage_derives_ready_resource_and_exact_assignment():
    policy = load_evidence_orchestration_config(POLICY_PATH)
    _profiles, requirements = _quiz_requirements()
    required = next(item for item in requirements if item.criticality == "required")
    supporting = next(item for item in requirements if item.criticality == "supporting")
    task = build_retrieval_task(
        requirement=required,
        source_type="local_rag",
        query="math assessable facts course notes",
        purpose=required.acceptance_criteria,
        priority="high",
        round_index=0,
        result_limit=policy.max_results_per_task,
    )
    source_identity = "1" * 64
    content_fingerprint = "2" * 64
    evidence_id = make_evidence_id(
        requirement_id=required.requirement_id,
        source_type="local_rag",
        source_identity_fingerprint=source_identity,
        content_fingerprint=content_fingerprint,
    )
    entry = EvidenceLedgerEntry(
        round_index=0,
        task_id=task.task_id,
        requirement_id=required.requirement_id,
        evidence_id=evidence_id,
        resource_type="quiz",
        subject="math",
        source_type="local_rag",
        candidate_ref="child_math_1",
        candidate_snapshot_fingerprint="3" * 64,
        source_identity_fingerprint=source_identity,
        content_fingerprint=content_fingerprint,
        accepted=True,
        rejection_reason_code="",
    )
    batch = RequirementCoverageBatch(
        schema_version="requirement_coverage_batch_v1",
        round_index=0,
        coverages=[
            RequirementCoverage(
                requirement_id=required.requirement_id,
                resource_type="quiz",
                subject="math",
                round_index=0,
                coverage_state="complete",
                evidence_ids=[evidence_id],
                confidence=0.9,
                reason="The accepted course excerpt supports the assessable facts.",
                suggested_local_query="",
                suggested_web_query="",
            ),
            RequirementCoverage(
                requirement_id=supporting.requirement_id,
                resource_type="quiz",
                subject="math",
                round_index=0,
                coverage_state="missing",
                evidence_ids=[],
                confidence=0.0,
                reason="No misconception-boundary evidence was retrieved.",
                suggested_local_query="math misconception boundaries course notes",
                suggested_web_query="math misconception boundaries official tutorial",
            ),
        ],
    )
    compiled_batch = compile_requirement_coverage_batch(batch)

    validate_evidence_ledger(
        entries=(entry,),
        tasks=(task,),
        requirements=requirements,
        config=policy,
    )
    validate_requirement_coverage(
        batch=compiled_batch,
        requirements=requirements,
        entries=(entry,),
    )
    readiness = derive_resource_readiness(
        requested_resource_types=("quiz",),
        requirements=requirements,
        batch=compiled_batch,
    )
    assignments = derive_resource_evidence_assignments(
        readiness=readiness,
        requirements=requirements,
        batch=compiled_batch,
        entries=(entry,),
    )

    assert readiness[0].readiness_state == "ready"
    assert readiness[0].blocked_requirement_ids == ()
    assert assignments[0].resource_type == "quiz"
    assert assignments[0].topic_ids == ("math.algebra",)
    assert assignments[0].evidence_ids == (evidence_id,)


def test_multi_resource_readiness_allows_only_ready_resource_assignment():
    policy = load_evidence_orchestration_config(POLICY_PATH)
    profiles = load_resource_evidence_profiles(PROFILES_PATH)
    quiz_need = next(
        need
        for need in profiles.profile_for("quiz").needs
        if need.criticality == "required"
    )
    mindmap_need = next(
        need
        for need in profiles.profile_for("mindmap").needs
        if need.criticality == "required"
    )
    drafts = EvidenceRequirementDraftBatch(
        schema_version="evidence_requirement_draft_batch_v1",
        requirements=[
            EvidenceRequirementDraft(
                resource_type="quiz",
                subject="math",
                topic_id="math.algebra",
                profile_need_id=quiz_need.need_id,
                evidence_kind=quiz_need.evidence_kind,
                scope=quiz_need.scope,
                criticality=quiz_need.criticality,
                source_policy=quiz_need.source_policy,
                acceptance_criteria=quiz_need.acceptance_criteria,
                query_intent="math quiz assessable evidence",
            ),
            EvidenceRequirementDraft(
                resource_type="mindmap",
                subject="math",
                topic_id="math.algebra",
                profile_need_id=mindmap_need.need_id,
                evidence_kind=mindmap_need.evidence_kind,
                scope=mindmap_need.scope,
                criticality=mindmap_need.criticality,
                source_policy=mindmap_need.source_policy,
                acceptance_criteria=mindmap_need.acceptance_criteria,
                query_intent="math concept relationship evidence",
            ),
        ],
    )
    quiz_requirement, mindmap_requirement = compile_evidence_requirement_batch(drafts)
    task = build_retrieval_task(
        requirement=quiz_requirement,
        source_type="local_rag",
        query="math quiz assessable facts course notes",
        purpose=quiz_requirement.acceptance_criteria,
        priority="high",
        round_index=0,
        result_limit=policy.max_results_per_task,
    )
    source_identity = "4" * 64
    content_fingerprint = "5" * 64
    evidence_id = make_evidence_id(
        requirement_id=quiz_requirement.requirement_id,
        source_type="local_rag",
        source_identity_fingerprint=source_identity,
        content_fingerprint=content_fingerprint,
    )
    entry = EvidenceLedgerEntry(
        round_index=0,
        task_id=task.task_id,
        requirement_id=quiz_requirement.requirement_id,
        evidence_id=evidence_id,
        resource_type="quiz",
        subject="math",
        source_type="local_rag",
        candidate_ref="child_math_quiz",
        candidate_snapshot_fingerprint="6" * 64,
        source_identity_fingerprint=source_identity,
        content_fingerprint=content_fingerprint,
        accepted=True,
        rejection_reason_code="",
    )
    coverage = RequirementCoverageBatch(
        schema_version="requirement_coverage_batch_v1",
        round_index=0,
        coverages=[
            RequirementCoverage(
                requirement_id=quiz_requirement.requirement_id,
                resource_type="quiz",
                subject="math",
                round_index=0,
                coverage_state="complete",
                evidence_ids=[evidence_id],
                confidence=0.9,
                reason="The course excerpt supports assessable quiz facts.",
                suggested_local_query="",
                suggested_web_query="",
            ),
            RequirementCoverage(
                requirement_id=mindmap_requirement.requirement_id,
                resource_type="mindmap",
                subject="math",
                round_index=0,
                coverage_state="missing",
                evidence_ids=[],
                confidence=0.0,
                reason="No concept relationship evidence was retrieved.",
                suggested_local_query="math concept relationships hierarchy",
                suggested_web_query="",
            ),
        ],
    )
    compiled_coverage = compile_requirement_coverage_batch(coverage)
    requirements = (quiz_requirement, mindmap_requirement)
    readiness = derive_resource_readiness(
        requested_resource_types=("quiz", "mindmap"),
        requirements=requirements,
        batch=compiled_coverage,
    )
    assignments = derive_resource_evidence_assignments(
        readiness=readiness,
        requirements=requirements,
        batch=compiled_coverage,
        entries=(entry,),
    )

    assert [item.readiness_state for item in readiness] == [
        "ready",
        "blocked_insufficient_evidence",
    ]
    assert [item.resource_type for item in assignments] == ["quiz"]


def test_coverage_reports_all_dual_source_query_shape_violations() -> None:
    requirements = _requirements_for_source_policies(
        "local_and_web",
        "local_and_web",
    )
    secret_queries = ("private local alpha", "private web beta")
    batch = compile_requirement_coverage_batch(
        RequirementCoverageBatch(
            schema_version="requirement_coverage_batch_v1",
            round_index=0,
            coverages=[
                RequirementCoverage(
                    requirement_id=requirements[0].requirement_id,
                    resource_type="quiz",
                    subject="math",
                    round_index=0,
                    coverage_state="missing",
                    evidence_ids=[],
                    confidence=0.0,
                    reason="The dual-source evidence is incomplete.",
                    suggested_local_query=secret_queries[0],
                    suggested_web_query="",
                ),
                RequirementCoverage(
                    requirement_id=requirements[1].requirement_id,
                    resource_type="quiz",
                    subject="math",
                    round_index=0,
                    coverage_state="missing",
                    evidence_ids=[],
                    confidence=0.0,
                    reason="The dual-source evidence is incomplete.",
                    suggested_local_query="",
                    suggested_web_query=secret_queries[1],
                ),
            ],
        )
    )

    with pytest.raises(RequirementCoverageValidationError) as exc_info:
        validate_requirement_coverage(
            batch=batch,
            requirements=requirements,
            entries=(),
        )

    assert exc_info.value.code == "invalid_dual_source_gap_query"
    assert exc_info.value.reason.count("source_policy=local_and_web") == 2
    assert "actual_shape=local_only" in exc_info.value.reason
    assert "actual_shape=web_only" in exc_info.value.reason
    assert exc_info.value.reason.count("required_shape=both") == 2
    ordered_ids = sorted(item.requirement_id for item in requirements)
    assert exc_info.value.reason.index(ordered_ids[0]) < exc_info.value.reason.index(
        ordered_ids[1]
    )
    assert all(query not in str(exc_info.value) for query in secret_queries)


def test_coverage_uses_matrix_code_for_mixed_query_shape_violations() -> None:
    requirements = _requirements_for_source_policies("local_only", "web_only")
    secret_queries = (
        "private mixed local alpha",
        "private mixed web alpha",
        "private mixed local beta",
        "private mixed web beta",
    )
    batch = compile_requirement_coverage_batch(
        RequirementCoverageBatch(
            schema_version="requirement_coverage_batch_v1",
            round_index=0,
            coverages=[
                RequirementCoverage(
                    requirement_id=requirement.requirement_id,
                    resource_type="quiz",
                    subject="math",
                    round_index=0,
                    coverage_state="missing",
                    evidence_ids=[],
                    confidence=0.0,
                    reason="The source-specific evidence is incomplete.",
                    suggested_local_query=secret_queries[index * 2],
                    suggested_web_query=secret_queries[index * 2 + 1],
                )
                for index, requirement in enumerate(requirements)
            ],
        )
    )

    with pytest.raises(RequirementCoverageValidationError) as exc_info:
        validate_requirement_coverage(
            batch=batch,
            requirements=requirements,
            entries=(),
        )

    assert exc_info.value.code == "invalid_source_gap_query_matrix"
    assert "source_policy=local_only,actual_shape=both" in exc_info.value.reason
    assert "source_policy=web_only,actual_shape=both" in exc_info.value.reason
    assert "required_shape=local_only" in exc_info.value.reason
    assert "required_shape=web_only" in exc_info.value.reason
    assert all(query not in str(exc_info.value) for query in secret_queries)


def test_coverage_reports_all_invalid_evidence_bindings_without_ids() -> None:
    requirements = _requirements_for_source_policies("local_only", "web_only")
    entries_list: list[EvidenceLedgerEntry] = []
    for index, requirement in enumerate(requirements):
        source_identity_fingerprint = f"{index + 11:064x}"
        content_fingerprint = f"{index + 21:064x}"
        evidence_id = make_evidence_id(
            requirement_id=requirement.requirement_id,
            source_type="local_rag" if index == 0 else "web",
            source_identity_fingerprint=source_identity_fingerprint,
            content_fingerprint=content_fingerprint,
        )
        entries_list.append(
            EvidenceLedgerEntry(
                round_index=0,
                task_id=f"task-{index}",
                requirement_id=requirement.requirement_id,
                evidence_id=evidence_id,
                resource_type=requirement.resource_type,
                subject=requirement.subject,
                source_type="local_rag" if index == 0 else "web",
                candidate_ref=f"candidate-{index}",
                candidate_snapshot_fingerprint=f"{index + 1:064x}",
                source_identity_fingerprint=source_identity_fingerprint,
                content_fingerprint=content_fingerprint,
                accepted=True,
                rejection_reason_code="",
            )
        )
    entries = tuple(entries_list)
    unknown_evidence_id = make_evidence_id(
        requirement_id=requirements[1].requirement_id,
        source_type="web",
        source_identity_fingerprint="9" * 64,
        content_fingerprint="a" * 64,
    )
    private_evidence_ids = (
        entries[0].evidence_id,
        entries[1].evidence_id,
        unknown_evidence_id,
    )
    batch = compile_requirement_coverage_batch(
        RequirementCoverageBatch(
            schema_version="requirement_coverage_batch_v1",
            round_index=0,
            coverages=[
                RequirementCoverage(
                    requirement_id=requirements[0].requirement_id,
                    resource_type="quiz",
                    subject="math",
                    round_index=0,
                    coverage_state="partial",
                    evidence_ids=[private_evidence_ids[1]],
                    confidence=0.5,
                    reason="The accepted evidence is incomplete.",
                    suggested_local_query="math private local repair",
                    suggested_web_query="",
                ),
                RequirementCoverage(
                    requirement_id=requirements[1].requirement_id,
                    resource_type="quiz",
                    subject="math",
                    round_index=0,
                    coverage_state="partial",
                    evidence_ids=[private_evidence_ids[2]],
                    confidence=0.5,
                    reason="The accepted evidence is incomplete.",
                    suggested_local_query="",
                    suggested_web_query="math private web repair",
                ),
            ],
        )
    )

    with pytest.raises(RequirementCoverageValidationError) as exc_info:
        validate_requirement_coverage(
            batch=batch,
            requirements=requirements,
            entries=entries,
        )

    assert exc_info.value.code == "invalid_coverage_evidence_ref"
    assert (
        f"requirement_id={requirements[0].requirement_id},unknown_ref_count=0,"
        "cross_requirement_ref_count=1"
    ) in exc_info.value.reason
    assert (
        f"requirement_id={requirements[1].requirement_id},unknown_ref_count=1,"
        "cross_requirement_ref_count=0"
    ) in exc_info.value.reason
    assert all(
        evidence_id not in str(exc_info.value) for evidence_id in private_evidence_ids
    )
