"""Real PostgreSQL checkpoint reconstruction acceptance test.

The test is skipped unless ``A3_TEST_POSTGRES_URI`` is explicitly provided.
It never calls an LLM and deletes its random thread checkpoint on completion.
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app import _thread_status_from_snapshot
from src.config import get_setting
from src.context_engineering.influence import (
    build_influence_entry,
    build_influence_update,
)
from src.context_engineering.input_manifest import (
    build_background_context_window,
    build_thread_context_ledger_update,
    llm_input_manifest_trace_payload,
)
from src.context_engineering.workspace import (
    build_workspace_artifact_update,
    build_workspace_evidence_update,
    merge_task_workspace,
)
from src.database.checkpointer import make_thread_config
from src.graph.builder import get_compiled_graph
from src.graph.qa import QAResponse, QASuggestion, build_qa_final_payload
from src.graph.resource_final import normalize_resource_final_payload
from src.observability.activity import build_activity_event
from src.observability.llm_input import build_llm_input_observation

_POSTGRES_URI_ENV = "A3_TEST_POSTGRES_URI"


def _required_postgres_uri() -> str:
    uri = str(os.getenv(_POSTGRES_URI_ENV) or "").strip()
    if not uri:
        pytest.skip(f"{_POSTGRES_URI_ENV} is required for PostgreSQL integration")
    return uri


def _configured_llm_identity(node_name: str) -> tuple[str, str]:
    provider = str(get_setting(f"llm.{node_name}.provider") or "").strip()
    model = str(get_setting(f"llm.{node_name}.model") or "").strip()
    if not provider or not model:
        raise RuntimeError(f"llm.{node_name} provider and model are required")
    return provider, model


def _durable_state(thread_id: str) -> tuple[dict, dict]:
    request_id = "phase7-request-1"
    base_state = {
        "thread_id": thread_id,
        "request_id": request_id,
        "subject": "Machine Learning",
        "primary_subject": "Machine Learning",
        "learning_goal": "Build reliable foundations",
    }
    evidence_update = build_workspace_evidence_update(
        base_state,
        {
            "evidence_judge_output": {
                "overall_evidence_state": "sufficient",
                "judged_evidence": [
                    {
                        "evidence_id": "local:phase7",
                        "keep": True,
                        "coverage_contribution": "Supports the learning sequence.",
                        "evidence_score": 0.9,
                    }
                ],
                "coverage_gaps": [
                    {
                        "gap": "More worked examples are useful.",
                        "priority": 0.5,
                    }
                ],
            },
            "graded_evidence": [
                {
                    "evidence_id": "local:phase7",
                    "title": "Approved course note",
                    "source_type": "local_rag",
                    "subject": "Machine Learning",
                    "coverage_contribution": "Supports the learning sequence.",
                    "evidence_score": 0.9,
                }
            ],
        },
    )
    workspace = merge_task_workspace({}, evidence_update)
    artifact_update = build_workspace_artifact_update(
        {**base_state, "task_workspace": workspace},
        [
            {
                "resource_type": "mindmap",
                "status": "success",
                "title": "Machine Learning Map",
                "artifact": {
                    "title": "Machine Learning Map",
                    "filename": "machine-learning-map.html",
                    "html_url": "/artifacts/mindmaps/phase7/map.html",
                },
                "message_preview": "Mindmap generated.",
                "metrics": {"node_count": 8},
            }
        ],
    )
    workspace = merge_task_workspace(workspace, artifact_update["task_workspace"])

    provider, model = _configured_llm_identity("mindmap")
    observation = build_llm_input_observation(
        node_name="mindmap_agent",
        llm_node="mindmap",
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": "compact integration prompt"}],
        state={**base_state, "task_workspace": workspace},
        call_purpose="structured_llm",
        output_mode="json_schema",
        schema_name="MindmapArtifact",
        schema_contract_first=True,
    )
    assert observation.context_usage_report is not None
    manifest = llm_input_manifest_trace_payload(observation.manifest)
    report = observation.context_usage_report.model_dump(mode="json")
    ledger = build_thread_context_ledger_update(existing={}, manifest=manifest)
    background = build_background_context_window(
        manifest=manifest,
        state={**base_state, "task_workspace": workspace},
        manifest_count=1,
        max_context_tokens=16_000,
    )
    influence = build_influence_entry(
        state=base_state,
        kind="planner_output",
        source_node="mindmap_planner",
        title="Mindmap plan",
        preview="Compact planner summary.",
    )
    influence_update = build_influence_update(
        state=base_state,
        entries=[influence],
    )
    activity = build_activity_event(
        thread_id=thread_id,
        request_id=request_id,
        sequence=1,
        kind="artifact",
        status="completed",
        activity_key="artifact:mindmap",
        title="Mindmap generated",
        summary="Renderable artifact finalized.",
        node="resource_bundle_output",
        safe_details={"resource_type": "mindmap"},
        now="2026-07-11T00:00:00+00:00",
    ).model_dump(mode="json")
    mindmap_final = normalize_resource_final_payload(
        {
            "type": "resource_final",
            "resource_type": "mindmap",
            "thread_id": thread_id,
            "request_id": request_id,
            "mindmap": {
                "title": "Machine Learning Map",
                "tree": {"title": "Machine Learning", "children": []},
            },
        }
    )
    assert mindmap_final is not None
    qa_final = build_qa_final_payload(
        response=QAResponse(
            answer="The capability registry is available.",
            uncertainty_note="",
            grounding_status="capability_registry",
            suggestions=[QASuggestion(label="Continue", action="continue_qa")],
        ),
        qa_scope="a3_agent",
        thread_id=thread_id,
        request_id=request_id,
    )
    durable = {
        **base_state,
        "conversation_summary": "Compact thread summary.",
        "evidence_summary_memory": [
            {
                "memory_id": "memory:evidence:phase7",
                "created_at": "2026-07-11T00:00:00+00:00",
                "summary": "Approved evidence summary.",
            }
        ],
        "evidence_gap_memory": [
            {
                "memory_id": "memory:gap:phase7",
                "created_at": "2026-07-11T00:00:00+00:00",
                "summary": "Worked examples remain useful.",
            }
        ],
        "task_workspace": workspace,
        "workspace_events": artifact_update["workspace_events"],
        "resource_artifacts_by_type": artifact_update["resource_artifacts_by_type"],
        "last_generated_artifacts": artifact_update["last_generated_artifacts"],
        "llm_input_manifest": manifest,
        "llm_input_manifests": [manifest],
        "thread_context_ledger": ledger,
        "background_context_window": background,
        "context_continuity": {
            "continuity_type": "task_continuity",
            "workspace_id": workspace["workspace_id"],
        },
        "context_influence_ledger": influence_update,
        "context_usage_report": report,
        "context_usage_reports": [report],
        "activity_timeline": [activity],
        "last_resource_final_payload": mindmap_final,
        "last_qa_response": qa_final,
    }
    study_plan_final = normalize_resource_final_payload(
        {
            "type": "resource_final",
            "resource_type": "study_plan",
            "thread_id": thread_id,
            "request_id": "phase7-request-2",
            "study_plan": {
                "title": "Machine Learning Study Plan",
                "phases": [{"title": "Foundations"}],
            },
        }
    )
    assert study_plan_final is not None
    return durable, study_plan_final


async def _assert_postgres_reconstruction() -> None:
    uri = _required_postgres_uri()
    thread_id = f"phase7-{uuid4()}"
    config = make_thread_config(thread_id)
    durable, latest_resource = _durable_state(thread_id)

    try:
        async with AsyncPostgresSaver.from_conn_string(uri) as saver:
            await saver.setup()
            graph = get_compiled_graph(checkpointer=saver)
            await graph.aupdate_state(config, durable)
            await graph.aupdate_state(
                config,
                {"last_resource_final_payload": latest_resource},
                as_node="resource_bundle_output",
            )

        async with AsyncPostgresSaver.from_conn_string(uri) as restored_saver:
            restored_graph = get_compiled_graph(checkpointer=restored_saver)
            restored_snapshot = await restored_graph.aget_state(config)
            values = restored_snapshot.values
            status = _thread_status_from_snapshot(thread_id, restored_snapshot)

        assert (
            values["task_workspace"]["workspace_id"]
            == durable["task_workspace"]["workspace_id"]
        )
        assert len(values["task_workspace"]["evidence_summaries"]) == 1
        assert len(values["task_workspace"]["artifacts_by_id"]) == 1
        assert len(values["workspace_events"]) == 1
        assert len(values["llm_input_manifests"]) == 1
        assert values["thread_context_ledger"]["manifest_count"] == 1
        assert values["background_context_window"]["workspace_present"] is True
        assert values["context_continuity"]["continuity_type"] == "task_continuity"
        assert values["context_influence_ledger"]["total_recorded"] == 1
        assert len(values["context_influence_ledger"]["entries_by_id"]) == 1
        assert len(values["context_usage_reports"]) == 1
        assert len(values["activity_timeline"]) == 1
        assert len(values["resource_artifacts_by_type"]) == 1
        assert len(values["last_generated_artifacts"]) == 1
        assert values["conversation_summary"] == "Compact thread summary."
        assert len(values["evidence_summary_memory"]) == 1
        assert len(values["evidence_gap_memory"]) == 1
        assert values["last_qa_response"]["type"] == "qa_final"
        assert values["last_resource_final_payload"]["resource_type"] == "study_plan"
        assert "mindmap" not in values["last_resource_final_payload"]
        assert set(values["last_resource_final_payload"]["resource"]["payload"]) == {
            "study_plan"
        }
        assert status.context_usage_report_count == 1
        assert status.llm_input_manifest_count == 1
        assert status.activity_timeline_count == 1
        assert status.background_context_window["workspace_present"] is True
        assert status.last_resource_final_payload["resource_type"] == "study_plan"
        assert status.last_qa_response["type"] == "qa_final"
    finally:
        async with AsyncPostgresSaver.from_conn_string(uri) as cleanup_saver:
            await cleanup_saver.adelete_thread(thread_id)


def test_postgres_rebuild_restores_durable_context_without_resource_leakage():
    loop = asyncio.SelectorEventLoop() if os.name == "nt" else asyncio.new_event_loop()
    try:
        loop.run_until_complete(_assert_postgres_reconstruction())
    finally:
        loop.close()
