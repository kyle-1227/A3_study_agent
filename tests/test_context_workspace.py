"""Tests for versioned Context Engineering task workspace helpers."""

from __future__ import annotations

from src.context_engineering.workspace import (
    ARTIFACT_ID_PREFIX,
    EVIDENCE_ID_PREFIX,
    GAP_ID_PREFIX,
    WORKSPACE_ID_PREFIX,
    build_workspace_artifact_update,
    build_workspace_evidence_update,
    merge_task_workspace,
    stable_artifact_id,
    stable_evidence_id,
    stable_gap_id,
    stable_workspace_id,
    workspace_continuation_context,
    workspace_continuation_trace_payload,
    workspace_status_payload,
    workspace_trace_payload,
)


def _state(**overrides):
    state = {
        "thread_id": "thread-1",
        "session_id": "thread-1",
        "request_id": "request-1",
        "subject": " Machine-Learning ",
        "learning_goal": "Understand gradient descent",
    }
    state.update(overrides)
    return state


def test_stable_ids_are_versioned_sha256_style():
    workspace_id = stable_workspace_id(
        thread_id="t",
        normalized_subject="math",
        normalized_learning_goal="learn_limits",
    )
    artifact_id = stable_artifact_id(
        resource_type="review_doc",
        title="Review",
        thread_id="t",
        request_id="r",
        normalized_subject="math",
        artifact_refs={"filename": "review.md"},
    )
    evidence_id = stable_evidence_id(
        original_evidence_id="local:1",
        thread_id="t",
        normalized_subject="math",
        request_id="r",
    )
    gap_id = stable_gap_id(
        gap="missing examples", subject="math", role="", thread_id="t"
    )

    assert workspace_id.startswith(f"{WORKSPACE_ID_PREFIX}:")
    assert artifact_id.startswith(f"{ARTIFACT_ID_PREFIX}:")
    assert evidence_id.startswith(f"{EVIDENCE_ID_PREFIX}:")
    assert gap_id.startswith(f"{GAP_ID_PREFIX}:")
    assert workspace_id == stable_workspace_id(
        thread_id="t",
        normalized_subject="math",
        normalized_learning_goal="learn_limits",
    )


def test_workspace_reducer_is_idempotent_and_bounds_sections():
    update = build_workspace_evidence_update(
        _state(),
        {
            "evidence_judge_output": {
                "overall_evidence_state": "sufficient",
                "judged_evidence": [
                    {
                        "evidence_id": "local:1",
                        "keep": True,
                        "coverage_contribution": "Explains the update rule.",
                        "evidence_score": 0.9,
                    }
                ],
                "coverage_gaps": [
                    {
                        "gap": "Need worked numerical examples.",
                        "suggested_search_query": "gradient descent worked example",
                        "priority": 0.8,
                    }
                ],
            },
            "graded_evidence": [
                {
                    "evidence_id": "local:1",
                    "title": "Course note",
                    "source_type": "local_rag",
                    "subject": "Machine Learning",
                    "coverage_contribution": "Explains the update rule.",
                    "evidence_score": 0.9,
                }
            ],
        },
    )

    once = merge_task_workspace({}, update)
    twice = merge_task_workspace(once, update)

    assert once["schema_version"] == 1
    assert len(twice["evidence_summaries"]) == 1
    assert len(twice["coverage_gaps"]) == 1
    assert twice["workspace_id"] == once["workspace_id"]
    assert workspace_status_payload(twice)["workspace_evidence_summary_count"] == 1


def test_workspace_rotates_on_clear_subject_change():
    first = merge_task_workspace({}, build_workspace_evidence_update(_state(), {}))
    second = merge_task_workspace(
        first,
        build_workspace_evidence_update(
            _state(subject="physics", learning_goal="Study Newton laws"),
            {},
        ),
    )

    assert second["workspace_id"] != first["workspace_id"]
    assert second.get("rotation_action") == "rotate"


def test_evidence_builder_uses_kept_evidence_only_and_gaps_separately():
    update = build_workspace_evidence_update(
        _state(),
        {
            "evidence_judge_output": {
                "overall_evidence_state": "partially_sufficient",
                "judged_evidence": [
                    {
                        "evidence_id": "keep-me",
                        "keep": True,
                        "coverage_contribution": "Kept contribution.",
                        "evidence_score": 0.7,
                    },
                    {
                        "evidence_id": "drop-me",
                        "keep": False,
                        "coverage_contribution": "",
                        "evidence_score": 0.1,
                    },
                ],
                "coverage_gaps": [
                    {
                        "gap": "Missing assessment examples.",
                        "suggested_search_query": "examples",
                    }
                ],
            },
            "graded_evidence": [
                {"evidence_id": "keep-me", "summary": "raw kept"},
                {"evidence_id": "drop-me", "summary": "raw rejected"},
            ],
        },
    )

    summaries = update["evidence_summaries"]
    assert len(summaries) == 1
    assert summaries[0]["original_evidence_id"] == "keep-me"
    assert "raw rejected" not in repr(update)
    assert len(update["coverage_gaps"]) == 1
    assert update["coverage_gaps"][0]["gap_id"].startswith(f"{GAP_ID_PREFIX}:")


def test_artifact_builder_keeps_safe_refs_and_rejects_unsafe_content():
    update = build_workspace_artifact_update(
        _state(),
        [
            {
                "resource_type": "review_doc",
                "status": "success",
                "title": "Gradient Review",
                "artifact": {
                    "title": "Gradient Review",
                    "markdown": "# huge raw body",
                    "filename": "review.md",
                    "markdown_url": "https://example.test/review.md?token=secret",
                    "docx_url": "https://example.test/review.docx?download=1",
                    "path": "artifacts/review/review.md",
                },
                "message_preview": "Generated review document.",
                "metrics": {"markdown_chars": 999},
            }
        ],
    )

    artifact = next(iter(update["task_workspace"]["artifacts_by_id"].values()))
    assert artifact["artifact_id"].startswith(f"{ARTIFACT_ID_PREFIX}:")
    assert artifact["artifact_refs"]["filename"] == "review.md"
    assert artifact["artifact_refs"]["docx_url"] == "https://example.test/review.docx"
    assert "markdown_url" not in artifact["artifact_refs"]
    assert "huge raw body" not in repr(update)


def test_trace_payload_has_only_counts_and_safe_metadata():
    update = build_workspace_evidence_update(
        _state(),
        {
            "evidence_judge_output": {
                "judged_evidence": [
                    {
                        "evidence_id": "e1",
                        "keep": True,
                        "coverage_contribution": "Secret sk-abc1234567890 text",
                        "evidence_score": 0.8,
                    }
                ]
            },
            "graded_evidence": [
                {
                    "evidence_id": "e1",
                    "coverage_contribution": "Secret sk-abc1234567890 text",
                }
            ],
        },
    )

    payload = workspace_trace_payload(update)

    assert payload["evidence_summary_count"] == 1
    assert "Secret" not in repr(payload)
    assert "sk-abc" not in repr(payload)


def test_corrupt_workspace_degrades_to_empty_status():
    status = workspace_status_payload(
        {"schema_version": 999, "evidence_summaries": "bad"}
    )

    assert status["workspace_present"] is False
    assert status["workspace_evidence_summary_count"] == 0


def test_workspace_continuation_allows_same_thread_resource_without_subject():
    workspace = merge_task_workspace(
        {},
        build_workspace_evidence_update(
            _state(subject="Machine Learning", learning_goal="Review core concepts"),
            {},
        ),
    )

    context = workspace_continuation_context(
        {
            "thread_id": "thread-1",
            "session_id": "thread-1",
            "request_id": "request-2",
            "subject": "other",
            "subject_candidates": [],
            "requested_resource_type": "mindmap",
            "requested_resource_types": ["mindmap"],
            "task_workspace": workspace,
        }
    )

    assert context["can_continue"] is True
    assert context["thread_id"] == "thread-1"
    assert context["current_thread_id"] == "thread-1"
    assert context["workspace_thread_id"] == "thread-1"
    assert context["normalized_subject"] == "machine_learning"
    assert context["active_learning_goal"] == "Review core concepts"


def test_workspace_continuation_trace_uses_current_thread_when_workspace_missing():
    context = workspace_continuation_context(
        {
            "session_id": "thread-1",
            "request_id": "request-2",
            "subject": "other",
            "subject_candidates": [],
            "requested_resource_type": "mindmap",
            "requested_resource_types": ["mindmap"],
        }
    )
    payload = workspace_continuation_trace_payload(context)

    assert context["can_continue"] is False
    assert context["skip_reason"] == "workspace_unavailable"
    assert payload["thread_id"] == "thread-1"
    assert payload["current_thread_id"] == "thread-1"
    assert payload["workspace_thread_id"] == ""


def test_workspace_continuation_skips_explicit_new_subject():
    workspace = merge_task_workspace(
        {},
        build_workspace_evidence_update(_state(subject="Machine Learning"), {}),
    )

    context = workspace_continuation_context(
        {
            "thread_id": "thread-1",
            "subject": "python",
            "subject_candidates": ["python"],
            "requested_resource_type": "mindmap",
            "requested_resource_types": ["mindmap"],
            "task_workspace": workspace,
        }
    )

    assert context["can_continue"] is False
    assert context["skip_reason"] == "current_subject_present"


def test_workspace_continuation_skips_thread_mismatch_and_corruption():
    workspace = merge_task_workspace(
        {},
        build_workspace_evidence_update(_state(subject="Machine Learning"), {}),
    )

    mismatch = workspace_continuation_context(
        {
            "thread_id": "thread-2",
            "requested_resource_type": "mindmap",
            "requested_resource_types": ["mindmap"],
            "task_workspace": workspace,
        }
    )
    corrupt = workspace_continuation_context(
        {
            "thread_id": "thread-1",
            "requested_resource_type": "mindmap",
            "requested_resource_types": ["mindmap"],
            "task_workspace": {"schema_version": 999},
        }
    )

    assert mismatch["can_continue"] is False
    assert mismatch["skip_reason"] == "thread_mismatch"
    mismatch_payload = workspace_continuation_trace_payload(mismatch)
    assert mismatch_payload["thread_id"] == "thread-2"
    assert mismatch_payload["workspace_thread_id"] == "thread-1"
    assert corrupt["can_continue"] is False
    assert corrupt["skip_reason"] == "workspace_unavailable"
