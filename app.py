"""A3 Study Agent - AI-powered university learning resource generation system."""

from __future__ import annotations

# ruff: noqa: E402
# The application loads .env before importing project modules that read settings at import time.

import asyncio
import hashlib
import json
import logging
import math
import os
import time
import uuid
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, NoReturn
from urllib.parse import unquote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
import psycopg
from pydantic import ValidationError

from src.assessment.attempt_contracts import (
    AssessmentAttemptV1,
    AssessmentCheckpointResourcesV2,
    stable_assessment_attempt_hash,
)
from src.assessment.attempt_journal import (
    AssessmentAttemptJournalError,
    AssessmentAttemptJournalV1,
    AssessmentAttemptRecordV1,
    AssessmentCheckpointIdempotencyExecutor,
    AssessmentExecutionLock,
    LocalAssessmentExecutionLock,
    find_assessment_attempt_record_v1,
    validate_assessment_attempt_journal_v1,
)
from src.assessment.attempt_service import (
    AssessmentAttemptService,
    AssessmentAttemptServiceError,
    AssessmentDependencyFailed,
    AssessmentIdentityError,
    AssessmentRecordedFailure,
    AssessmentRecoveryRequired,
    AssessmentRequestConflict,
)
from src.assessment.checkpoint import (
    AssessmentCheckpointError,
    validate_assessment_checkpoint_resources_v2,
)
from src.assessment.runtime import (
    classify_assessment_error_v1,
    generate_adaptive_practice_v1,
)
from src.context_engineering.schema import sanitize_error_message
from src.context_engineering.compaction import (
    CompactionResultV1,
    ProviderBoundUsageV1,
    build_compact_boundary,
    evaluate_full_compaction,
    get_full_compaction_config,
    provider_bound_usage_from_trace,
    summary_fingerprint,
)
from src.context_engineering.input_manifest import (
    background_context_status_payload,
    build_background_context_window,
    build_thread_context_ledger_update,
    llm_input_manifest_trace_payload,
    merge_llm_input_manifest_history,
)
from src.context_engineering.influence import influence_status_payload
from src.context_engineering.policy_mode import validate_context_runtime_policy
from src.context_engineering.session_memory import (
    ContextInjectionRecordV1,
    SessionContextMemoryLedgerV1,
    apply_context_memory_compaction,
    new_session_context_memory_ledger,
    record_context_injection,
)
from src.context_engineering.model_view import build_model_view_projection
from src.context_engineering.thread_window_v3 import build_thread_context_window_v3
from src.context_engineering.workspace import workspace_status_payload

load_dotenv(Path(__file__).parent / ".env")

from src.database.checkpointer import (
    checkpointer_enabled,
    checkpointer_type,
    get_db_uri,
    make_thread_config,
)
from src.database.assessment_lock import PostgresAssessmentExecutionLock
from src.config import get_setting
from src.graph.builder import get_compiled_resource_evidence_parent_child_graph
from src.graph.evidence_orchestration import EvidenceOrchestrationRuntime
from src.graph.served_candidate import (
    ServedCandidateRuntime,
    load_served_candidate_runtime,
)
from src.graph.state import (
    ACTIVITY_TIMELINE_CLEAR,
    CONTEXT_CLEAR,
    CONTEXT_INFLUENCE_LEDGER_CLEAR,
    CONTEXT_USAGE_REPORTS_CLEAR,
    DICT_CLEAR,
    GENERATED_ARTIFACTS_CLEAR,
    LLM_INPUT_MANIFESTS_CLEAR,
    MEMORY_CLEAR,
    SESSION_CONTEXT_MEMORY_LEDGER_CLEAR,
    TASK_WORKSPACE_CLEAR,
    WORKSPACE_EVENTS_CLEAR,
    initial_request_reset_transient_state,
)
from src.graph.resource_final_runtime import (
    requested_resource_kinds_from_state,
    resource_final_v3_required,
)
from src.graph.resource_final_v3 import validate_resource_final_v3
from src.graph.qa import qa_final_payload, qa_final_trace_payload
from src.learning_guidance.factory import load_learning_guidance_runtime
from src.learning_guidance.history_writer import (
    LearningGuidanceHistoryWriterError,
    LearningGuidanceHistoryWriterV1,
)
from src.learning_guidance.profile_writer import (
    LearningGuidanceProfileWriterV1,
    ProfileWriterError,
)
from src.learning_guidance.recommendation_final import (
    RecommendationFinalV1,
    validate_recommendation_final_v1,
)
from src.llm.compaction import invoke_conversation_compaction
from src.memory.storage import SQLiteMemoryStore
from src.profile import get_profile_manager, reset_profile_manager
from src.profile.schema import (
    AgentObservation,
    Goal,
    LearningStyle,
    SkillEntry,
    UserProfile,
)
from src.learning_guidance.runtime import LearningGuidanceRuntime
from src.profile.storage import SQLiteProfileStore
from src.run_control import (
    RUN_CONTROL_FIELDS,
    RUN_CONTROL_SCHEMA_VERSION,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_CONTINUING,
    RUN_STATUS_ERROR,
    RUN_STATUS_IDLE,
    RUN_STATUS_NOT_RESUMABLE,
    RUN_STATUS_RUNNING,
    RUN_STATUS_STOPPED,
    RUN_STATUS_STOPPING,
    RUN_STATUS_UNKNOWN,
    finish_active_run,
    get_active_run,
    run_control_registry,
    start_active_run,
    trim_context_usage_history,
    update_active_run,
    utc_now_iso,
)
from src.schemas import (
    ChatRequest,
    CompiledOnboardRequestV2,
    ContinueRequest,
    HealthLiveV1,
    HealthReadyV1,
    LearningGuidanceCatalogV1,
    OnboardRequest,
    OnboardResultV2,
    ResumeRequest,
    StopRequest,
    ThreadStatusResponse,
    compile_onboard_request_v2,
)
from src.observability.a3_trace import (
    emit_a3_trace,
    reset_trace_event_sink,
    set_trace_event_sink,
)
from src.observability.activity import (
    activity_from_trace_event,
    activity_timeline_status,
    build_activity_event,
    build_node_activity_event,
    merge_activity_timeline,
    next_activity_sequence,
)
from src.observability.context_usage_report import (
    merge_context_usage_report_history,
)
from src.observability.contracts import (
    ActivityEvent,
    ActivityStatus,
    ContextUsageReport,
    GraphManifest,
)
from src.observability.evidence_trace import is_evidence_trace_stage
from src.observability.graph_manifest import (
    build_graph_manifest,
    get_physical_graph_node_ids,
    graph_manifest_error_payload,
    graph_manifest_ref_payload,
    graph_manifest_status_payload,
)
from src.observability.node_registry import get_node_runtime_metadata
from src.observability.checkpointer_proxy import observe_checkpointer
from src.observability.performance_config import (
    load_performance_observability_config,
)
from src.observability.performance_contracts import FrontendPerformanceBatchV1
from src.observability.performance_service import (
    FrontendPerformanceRejected,
    configure_performance_service,
    current_frontend_performance_capability,
    get_performance_service,
    observe_request_performance,
)
from src.streaming.provisional import (
    reset_provisional_event_sink,
    set_provisional_event_sink,
)
from src.streaming.contracts import (
    AgentStreamDraftType,
    AgentStreamEventDraftV2,
    ContentBlockPayloadV1,
    StreamContractError,
)
from src.streaming.evidence_progress import (
    EvidenceProgressV1,
    reset_evidence_progress_sink,
    set_evidence_progress_sink,
)
from src.streaming.journal import StreamJournalExpiredError
from src.streaming.session import (
    StreamSessionConflictError,
    StreamSessionExpiredError,
    StreamSessionManager,
    StreamSessionNotFoundError,
)
from src.streaming.settings import load_streaming_runtime_config
from src.streaming.sse import parse_last_event_id
from src.tools.document_tool import (
    get_code_practice_artifact_dir,
    get_exercise_artifact_dir,
    get_review_doc_artifact_dir,
    get_video_script_artifact_dir,
)
from src.tools.mindmap_tool import get_mindmap_artifact_dir
from src.tools.video_animation_tool import get_video_animation_artifact_dir
from src.tracing import setup_tracing, shutdown_tracing

logger = logging.getLogger(__name__)
STREAMING_RUNTIME_CONFIG = load_streaming_runtime_config()
stream_session_manager = StreamSessionManager(STREAMING_RUNTIME_CONFIG)
FRONTEND_PERFORMANCE_ENDPOINT_PATH = (
    load_performance_observability_config().frontend_ingestion.endpoint_path
)
PROVIDER_RETRY_TRACE_STAGES = {
    "provider_transport_retry_attempt",
    "provider_transport_error",
    "final_failure_after_retries",
}
WORKSPACE_TRACE_STAGES = {
    "task_workspace.update_planned",
    "task_workspace.updated",
    "task_workspace.update_failed",
    "task_workspace.continuation_checked",
    "task_workspace.continuation_applied",
    "task_workspace.continuation_skipped",
    "resource_artifacts.indexed",
    "workspace_context.collected",
}


def _stream_draft(
    event_type: AgentStreamDraftType,
    data: dict | None = None,
) -> AgentStreamEventDraftV2:
    return AgentStreamEventDraftV2(type=event_type, data=dict(data or {}))


def _activity_update_draft(
    kind: str,
    payload: dict | None = None,
) -> AgentStreamEventDraftV2:
    return _stream_draft(
        "activity_update",
        {"kind": kind, "payload": dict(payload or {})},
    )


def _tool_progress_draft(
    kind: str,
    payload: dict | None = None,
) -> AgentStreamEventDraftV2:
    return _stream_draft(
        "tool_progress",
        {"kind": kind, "payload": dict(payload or {})},
    )


def _artifact_progress_draft(
    kind: str,
    payload: dict | None = None,
) -> AgentStreamEventDraftV2:
    return _stream_draft(
        "artifact_progress",
        {"kind": kind, "payload": dict(payload or {})},
    )


def _trace_progress_draft(payload: dict) -> AgentStreamEventDraftV2:
    """Convert one sanitized trace projection into its native public category."""

    kind = str(payload.get("type") or "trace_update")
    data = {key: value for key, value in payload.items() if key != "type"}
    if kind in {"provider_retry", "resource_subnode"}:
        return _tool_progress_draft(kind, data)
    if kind in {"resource_generation", "artifact"}:
        return _artifact_progress_draft(kind, data)
    return _activity_update_draft(kind, data)


def _content_block_draft(
    event_type: AgentStreamDraftType,
    *,
    block_id: str,
    block_index: int,
    delta: str = "",
    reset: bool = False,
    reason: str = "",
) -> AgentStreamEventDraftV2:
    payload = ContentBlockPayloadV1(
        block_id=block_id,
        block_index=block_index,
        block_type="markdown",
        provisional=True,
        delta=delta,
        reset=reset,
        reason=reason,
    )
    return _stream_draft(event_type, payload.model_dump(mode="json"))


CONTEXT_TOP_ITEM_FIELDS = {
    "id",
    "source_type",
    "title",
    "token_estimate",
    "priority",
    "scope",
    "lifetime",
    "disclosure_level",
}
PACKING_PREVIEW_FIELDS = {
    "id",
    "source_type",
    "title",
    "token_estimate",
    "priority",
    "can_drop",
    "reason",
}


def _safe_context_top_items(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    safe_items: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        safe_items.append(
            {key: item[key] for key in CONTEXT_TOP_ITEM_FIELDS if key in item}
        )
    return safe_items


def _safe_packing_preview_items(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    safe_items: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        safe_item = {key: item[key] for key in PACKING_PREVIEW_FIELDS if key in item}
        if "title" in safe_item:
            safe_item["title"] = sanitize_error_message(
                safe_item["title"],
                max_chars=120,
            )
        safe_items.append(safe_item)
    return safe_items


def _safe_int_dict(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    safe: dict = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        safe[sanitize_error_message(key, max_chars=80)] = item
    return safe


def _safe_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(value, 0)


def _safe_warning_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [sanitize_error_message(warning) for warning in value]


def _safe_reason_dict(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        sanitize_error_message(key, max_chars=80): sanitize_error_message(
            item,
            max_chars=120,
        )
        for key, item in value.items()
        if str(key or "").strip() and str(item or "").strip()
    }


def _safe_context_event_summary(event: dict) -> dict:
    return {
        "stage": sanitize_error_message(event.get("stage", ""), max_chars=120),
        "request_id": sanitize_error_message(
            event.get("request_id", ""),
            max_chars=80,
        ),
        "node": sanitize_error_message(event.get("node_name", ""), max_chars=120),
        "llm_node": sanitize_error_message(event.get("llm_node", ""), max_chars=120),
        "trace_call_id": sanitize_error_message(
            event.get("trace_call_id", ""),
            max_chars=80,
        ),
        "trace_seq": event.get("trace_seq", 0)
        if isinstance(event.get("trace_seq"), int)
        and not isinstance(event.get("trace_seq"), bool)
        else 0,
        "entry_count": _safe_int(event.get("entry_count"), default=0),
        "token_estimate": _safe_int(event.get("token_estimate"), default=0),
        "counts_by_kind": {
            sanitize_error_message(key, max_chars=80): _safe_int(value, default=0)
            for key, value in (event.get("counts_by_kind") or {}).items()
        }
        if isinstance(event.get("counts_by_kind"), dict)
        else {},
        "influence_ids": _safe_warning_list(event.get("influence_ids")),
    }


def _safe_workspace_event_summary(event: dict) -> dict:
    return {
        "stage": sanitize_error_message(event.get("stage", ""), max_chars=120),
        "request_id": sanitize_error_message(
            event.get("request_id", ""),
            max_chars=80,
        ),
        "thread_id": sanitize_error_message(
            event.get("thread_id", ""),
            max_chars=120,
        ),
        "current_thread_id": sanitize_error_message(
            event.get("current_thread_id", ""),
            max_chars=120,
        ),
        "workspace_thread_id": sanitize_error_message(
            event.get("workspace_thread_id", ""),
            max_chars=120,
        ),
        "workspace_id": sanitize_error_message(
            event.get("workspace_id", ""),
            max_chars=160,
        ),
        "active_subject": sanitize_error_message(
            event.get("active_subject", ""),
            max_chars=120,
        ),
        "active_learning_goal_present": bool(
            event.get("active_learning_goal_present", False)
        ),
        "evidence_summary_count": _safe_int(
            event.get("evidence_summary_count"),
            default=0,
        ),
        "coverage_gap_count": _safe_int(event.get("coverage_gap_count"), default=0),
        "artifact_count": _safe_int(event.get("artifact_count"), default=0),
        "constraint_count": _safe_int(event.get("constraint_count"), default=0),
        "updated_sources": _safe_warning_list(event.get("updated_sources")),
        "rotation_action": sanitize_error_message(
            event.get("rotation_action", ""),
            max_chars=80,
        ),
        "can_continue": bool(event.get("can_continue", False)),
        "continuation_applied": bool(event.get("continuation_applied", False)),
        "skip_reason": sanitize_error_message(
            event.get("skip_reason", ""),
            max_chars=120,
        ),
        "normalized_subject": sanitize_error_message(
            event.get("normalized_subject", ""),
            max_chars=120,
        ),
        "diagnostics": _safe_warning_list(event.get("diagnostics")),
    }


def _trace_common_payload(event: dict, *, event_type: str) -> dict:
    return {
        "type": event_type,
        "node": sanitize_error_message(event.get("node_name", ""), max_chars=120),
        "llm_node": sanitize_error_message(event.get("llm_node", ""), max_chars=120),
        "trace_call_id": sanitize_error_message(
            event.get("trace_call_id", ""),
            max_chars=80,
        ),
        "trace_seq": event.get("trace_seq", 0)
        if isinstance(event.get("trace_seq"), int)
        and not isinstance(event.get("trace_seq"), bool)
        else 0,
    }


def _context_policy_resolved_payload(event: dict) -> dict:
    payload = _trace_common_payload(event, event_type="context_policy_resolved")
    payload.update(
        {
            "mode": sanitize_error_message(event.get("mode", ""), max_chars=80),
            "risk_tier": event.get("risk_tier", 0),
            "policy_source": sanitize_error_message(
                event.get("policy_source", ""),
                max_chars=80,
            ),
            "required_sources": _safe_warning_list(event.get("required_sources")),
            "optional_sources": _safe_warning_list(event.get("optional_sources")),
            "injectable_sources": _safe_warning_list(event.get("injectable_sources")),
        }
    )
    return payload


def _context_provider_supply_plan_payload(event: dict) -> dict:
    payload = _trace_common_payload(event, event_type="context_provider_supply_plan")
    payload.update(
        {
            "requested_sources": _safe_warning_list(event.get("requested_sources")),
            "required_sources": _safe_warning_list(event.get("required_sources")),
            "optional_sources": _safe_warning_list(event.get("optional_sources")),
            "enabled_sources": _safe_warning_list(event.get("enabled_sources")),
            "disabled_sources": _safe_warning_list(event.get("disabled_sources")),
            "unregistered_sources": _safe_warning_list(
                event.get("unregistered_sources")
            ),
            "provider_count": event.get("provider_count", 0),
            "provider_sources_missing": _safe_int_dict(
                event.get("provider_sources_missing")
            ),
            "provider_missing_reasons": _safe_reason_dict(
                event.get("provider_missing_reasons")
            ),
        }
    )
    return payload


def _context_provider_supply_payload(event: dict) -> dict:
    payload = _trace_common_payload(event, event_type="context_provider_supply")
    payload.update(
        {
            "provider_count": event.get("provider_count", 0),
            "item_count": event.get("item_count", 0),
            "source_counts": _safe_int_dict(event.get("source_counts")),
            "provider_sources_missing": _safe_int_dict(
                event.get("provider_sources_missing")
            ),
            "provider_missing_reasons": _safe_reason_dict(
                event.get("provider_missing_reasons")
            ),
            "provider_error_count": event.get("provider_error_count", 0),
            "evidence_rejected_count": event.get("evidence_rejected_count", 0),
            "evidence_reject_reasons": _safe_int_dict(
                event.get("evidence_reject_reasons")
            ),
        }
    )
    return payload


def _context_source_filter_payload(event: dict) -> dict:
    payload = _trace_common_payload(event, event_type="context_source_filter")
    payload.update(
        {
            "source_counts_before": _safe_int_dict(event.get("source_counts_before")),
            "source_counts_after": _safe_int_dict(event.get("source_counts_after")),
            "source_counts_dropped": _safe_int_dict(event.get("source_counts_dropped")),
            "drop_reasons": _safe_int_dict(event.get("drop_reasons")),
            "source_drop_reasons": _safe_int_dict(event.get("source_drop_reasons")),
            "budget_drop_reasons": _safe_int_dict(event.get("budget_drop_reasons")),
            "warnings": _safe_warning_list(event.get("warnings")),
        }
    )
    return payload


def _graph_checkpointer_type(graph) -> str:
    configured_type = getattr(graph, "_a3_checkpointer_type", "")
    configured_enabled = getattr(graph, "_a3_checkpointer_enabled", None)
    if configured_enabled is False:
        return "none"
    if configured_type:
        return str(configured_type)
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        return "none"
    return type(checkpointer).__name__


def _emit_graph_config_trace(graph, config: dict, state: dict) -> None:
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    checkpointer_type = _graph_checkpointer_type(graph)
    # TEMP A3_TRACE: remove after state snapshot validation.
    emit_a3_trace(
        logger,
        "graph_config",
        {
            "checkpointer_enabled": checkpointer_type != "none",
            "checkpointer_type": checkpointer_type,
            "thread_id": configurable.get("thread_id", ""),
            "has_thread_id": bool(configurable.get("thread_id")),
        },
        state=state,
        env_flag="LOG_A3_TRACE",
    )


def _readiness_db_timeout_seconds() -> float:
    value = get_setting("server.readiness_db_timeout_seconds")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise RuntimeError(
            "server.readiness_db_timeout_seconds must be an explicit positive number"
        )
    return float(value)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage async resources: tracing, PostgreSQL checkpointer, graph."""
    setup_tracing()
    performance_config = load_performance_observability_config()
    performance_service = configure_performance_service(performance_config)
    app.state.performance_observability_enabled = performance_config.enabled
    app.state.frontend_performance_ingestion_enabled = (
        performance_config.frontend_ingestion.enabled
    )
    app.state.performance_service = performance_service
    app.state.readiness_db_timeout_seconds = _readiness_db_timeout_seconds()
    runtime_context_policy = validate_context_runtime_policy()
    app.state.context_policy_mode = runtime_context_policy.mode
    app.state.context_policy_environment = runtime_context_policy.environment
    emit_a3_trace(
        logger,
        "context_runtime_policy.validated",
        {
            "policy_mode": runtime_context_policy.mode,
            "environment": runtime_context_policy.environment,
            "max_items_total": runtime_context_policy.max_items_total,
            "max_injected_context_tokens": (
                runtime_context_policy.max_injected_context_tokens
            ),
            "enabled_sources": list(runtime_context_policy.enabled_sources),
            "eligible_node_roles": list(runtime_context_policy.eligible_node_roles),
        },
        state={},
        env_flag="LOG_A3_TRACE",
    )

    async with AsyncExitStack() as stack:
        checkpointer = None
        enabled = checkpointer_enabled()
        ckp_type = checkpointer_type()
        db_uri = get_db_uri()

        if enabled and ckp_type == "postgres":
            if not db_uri:
                raise RuntimeError(
                    "PostgreSQL checkpointer requires DB_URI when CHECKPOINTER_TYPE=postgres"
                )
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            checkpointer = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(db_uri)
            )
            await checkpointer.setup()
            logger.info("PostgreSQL checkpointer initialized")
        elif enabled and ckp_type == "memory":
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()
            ckp_type = "memory"
        elif enabled:
            raise RuntimeError(
                f"Unsupported LangGraph checkpointer type: {sanitize_error_message(ckp_type, max_chars=80)}"
            )
        else:
            logger.warning("LangGraph checkpointer disabled by configuration")
            ckp_type = "disabled"

        graph_checkpointer = observe_checkpointer(checkpointer)
        app.state.checkpointer_enabled = bool(checkpointer)
        app.state.checkpointer_type = ckp_type
        project_root = Path(__file__).resolve().parent
        profile_db_path = project_root / "data" / "profile.db"
        memory_db_path = project_root / "data" / "memory.db"
        memory_store = SQLiteMemoryStore(memory_db_path)
        await memory_store.initialize()
        learning_guidance_runtime = load_learning_guidance_runtime(
            config_path=project_root / "config" / "learning_guidance.yaml",
            project_root=project_root,
            profile_db_path=profile_db_path,
            memory_db_path=memory_db_path,
            clock=lambda: datetime.now(timezone.utc),
        )
        candidate_generation_id = os.environ["PARENT_CHILD_GENERATION_ID"]
        if (
            not candidate_generation_id
            or candidate_generation_id != candidate_generation_id.strip()
        ):
            raise RuntimeError(
                "PARENT_CHILD_GENERATION_ID must be a non-blank stripped identifier"
            )
        served_candidate = load_served_candidate_runtime(
            project_root=project_root,
            generation_id=candidate_generation_id,
            learning_guidance=learning_guidance_runtime,
            index_config_path=(
                project_root
                / "config"
                / "rag"
                / "index.production-candidate.inactive.yaml"
            ),
            index_root=project_root / "indexes" / "parent_child",
            policy_config_path=(
                project_root / "config" / "rag" / "evidence_orchestration.yaml"
            ),
            profiles_config_path=(
                project_root / "config" / "rag" / "resource_evidence_profiles.yaml"
            ),
            rollout_config_path=(project_root / "config" / "rag" / "rollout.yaml"),
        )
        stack.callback(served_candidate.close)
        graph = get_compiled_resource_evidence_parent_child_graph(
            served_candidate.orchestration,
            checkpointer=graph_checkpointer,
        )
        app.state.served_candidate_owner = served_candidate
        app.state.served_candidate_runtime = served_candidate.orchestration
        app.state.parent_child_generation_id = candidate_generation_id
        app.state.parent_child_generation_manifest_fingerprint = (
            served_candidate.generation_manifest_fingerprint
        )
        app.state.learning_guidance_runtime = learning_guidance_runtime
        app.state.learning_guidance_history_writer = LearningGuidanceHistoryWriterV1(
            store=memory_store,
            knowledge_graph=learning_guidance_runtime.knowledge_graph,
        )
        profile_store = SQLiteProfileStore(profile_db_path)
        profile_writer = LearningGuidanceProfileWriterV1(
            store=profile_store,
            knowledge_graph=learning_guidance_runtime.knowledge_graph,
        )
        app.state.learning_guidance_profile_writer = profile_writer
        reset_profile_manager()
        app.state.profile_manager = get_profile_manager(
            store=profile_store,
            guidance_writer=profile_writer,
        )
        runtime_node_ids = frozenset(get_physical_graph_node_ids(graph))
        setattr(graph, "_a3_node_ids", runtime_node_ids)
        setattr(graph, "_a3_checkpointer_enabled", bool(checkpointer))
        setattr(graph, "_a3_checkpointer_type", ckp_type)
        try:
            graph_manifest = build_graph_manifest(
                graph,
                context_policy_mode=runtime_context_policy.mode,
                checkpointer_enabled=bool(checkpointer),
                checkpointer_type=ckp_type,
            )
        except Exception as exc:
            manifest_error = graph_manifest_error_payload(exc)
            app.state.graph_manifest = None
            app.state.graph_manifest_error = manifest_error.model_dump(mode="json")
            app.state.graph_version = ""
            setattr(graph, "_a3_activity_events_enabled", False)
            emit_a3_trace(
                logger,
                "graph_manifest.failed",
                app.state.graph_manifest_error,
                state={},
                env_flag="LOG_A3_TRACE",
            )
            logger.exception("Graph manifest construction failed")
        else:
            app.state.graph_manifest = graph_manifest
            app.state.graph_manifest_error = None
            app.state.graph_version = graph_manifest.graph_version
            setattr(graph, "_a3_graph_version", graph_manifest.graph_version)
            setattr(graph, "_a3_activity_events_enabled", True)
            emit_a3_trace(
                logger,
                "graph_manifest.built",
                {
                    "schema_version": graph_manifest.schema_version,
                    "graph_version": graph_manifest.graph_version,
                    "node_count": len(graph_manifest.nodes),
                    "edge_count": len(graph_manifest.edges),
                },
                state={},
                env_flag="LOG_A3_TRACE",
            )
        app.state.graph = graph
        assessment_execution_lock: AssessmentExecutionLock | None = None
        if ckp_type == "postgres":
            if not db_uri:
                raise RuntimeError("PostgreSQL assessment execution requires DB_URI")
            assessment_execution_lock = PostgresAssessmentExecutionLock(db_uri)
        elif ckp_type == "memory":
            assessment_execution_lock = LocalAssessmentExecutionLock()
        app.state.assessment_attempt_service = (
            _build_assessment_attempt_service(
                graph,
                execution_lock=assessment_execution_lock,
            )
            if assessment_execution_lock is not None
            else None
        )
        yield

    shutdown_tracing()


app = FastAPI(title="A3 Study Agent API", lifespan=lifespan)


def _readiness_unavailable(code: str) -> NoReturn:
    raise HTTPException(status_code=503, detail=code)


def _ready_state_value(state: object, name: str, *, error_code: str) -> object:
    value = getattr(state, name, None)
    if value is None:
        _readiness_unavailable(error_code)
    return value


async def _probe_postgres_readiness(
    *,
    db_uri: str,
    timeout_seconds: float,
) -> None:
    try:
        async with asyncio.timeout(timeout_seconds):
            async with await psycopg.AsyncConnection.connect(db_uri) as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    row = await cursor.fetchone()
    except (TimeoutError, OSError, ValueError, psycopg.Error):
        _readiness_unavailable("health_ready_database_unavailable")
    if row != (1,):
        _readiness_unavailable("health_ready_database_unavailable")


@app.get("/health/live", response_model=HealthLiveV1)
async def health_live_endpoint() -> HealthLiveV1:
    return HealthLiveV1(schema_version="health_live_v1", status="live")


@app.get("/health/ready", response_model=HealthReadyV1)
async def health_ready_endpoint(request: Request) -> HealthReadyV1:
    state = request.app.state
    if getattr(state, "checkpointer_enabled", None) is not True:
        _readiness_unavailable("health_ready_postgres_checkpointer_required")
    if getattr(state, "checkpointer_type", None) != "postgres":
        _readiness_unavailable("health_ready_postgres_checkpointer_required")

    timeout_seconds = _ready_state_value(
        state,
        "readiness_db_timeout_seconds",
        error_code="health_ready_timeout_config_unavailable",
    )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        _readiness_unavailable("health_ready_timeout_config_invalid")

    manifest = _ready_state_value(
        state,
        "graph_manifest",
        error_code="health_ready_graph_manifest_unavailable",
    )
    graph_version = _ready_state_value(
        state,
        "graph_version",
        error_code="health_ready_graph_manifest_unavailable",
    )
    if (
        not isinstance(manifest, GraphManifest)
        or not isinstance(graph_version, str)
        or not graph_version
        or graph_version != manifest.graph_version
    ):
        _readiness_unavailable("health_ready_graph_manifest_invalid")

    guidance = _ready_state_value(
        state,
        "learning_guidance_runtime",
        error_code="health_ready_knowledge_graph_unavailable",
    )
    if not isinstance(guidance, LearningGuidanceRuntime):
        _readiness_unavailable("health_ready_knowledge_graph_invalid")
    knowledge_graph = guidance.knowledge_graph

    orchestration = _ready_state_value(
        state,
        "served_candidate_runtime",
        error_code="health_ready_orchestration_unavailable",
    )
    candidate_owner = _ready_state_value(
        state,
        "served_candidate_owner",
        error_code="health_ready_generation_manifest_unavailable",
    )
    if (
        not isinstance(orchestration, EvidenceOrchestrationRuntime)
        or not isinstance(candidate_owner, ServedCandidateRuntime)
        or candidate_owner.orchestration is not orchestration
        or orchestration.learning_guidance is not guidance
    ):
        _readiness_unavailable("health_ready_orchestration_invalid")

    generation_id = _ready_state_value(
        state,
        "parent_child_generation_id",
        error_code="health_ready_generation_unavailable",
    )
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or generation_id != generation_id.strip()
        or generation_id != orchestration.parent_child.generation_id
    ):
        _readiness_unavailable("health_ready_generation_invalid")
    generation_manifest_fingerprint = _ready_state_value(
        state,
        "parent_child_generation_manifest_fingerprint",
        error_code="health_ready_generation_manifest_unavailable",
    )
    if (
        not isinstance(generation_manifest_fingerprint, str)
        or generation_manifest_fingerprint
        != candidate_owner.generation_manifest_fingerprint
    ):
        _readiness_unavailable("health_ready_generation_manifest_invalid")

    db_uri = get_db_uri()
    if not isinstance(db_uri, str) or not db_uri or db_uri != db_uri.strip():
        _readiness_unavailable("health_ready_database_config_unavailable")
    await _probe_postgres_readiness(
        db_uri=db_uri,
        timeout_seconds=float(timeout_seconds),
    )

    try:
        return HealthReadyV1(
            schema_version="health_ready_v1",
            status="ready",
            checkpointer_type="postgres",
            graph_version=graph_version,
            knowledge_graph_data_version=knowledge_graph.data_version,
            knowledge_graph_artifact_fingerprint=(knowledge_graph.artifact_fingerprint),
            parent_child_generation_id=generation_id,
            parent_child_generation_manifest_fingerprint=(
                generation_manifest_fingerprint
            ),
            evidence_orchestration_fingerprint=(
                orchestration.orchestration_fingerprint
            ),
            candidate_mode="inactive_canary",
        )
    except ValidationError:
        _readiness_unavailable("health_ready_identity_invalid")


class AgentEventStreamResponse(StreamingResponse):
    """Document and deliver the public agent_stream_v2 media type."""

    media_type = "text/event-stream"


_SSE_OPENAPI_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "agent_stream_v2 server-sent events",
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    }
}

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

FastAPIInstrumentor.instrument_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _runtime_graph_node_ids(graph) -> frozenset[str]:
    configured = getattr(graph, "_a3_node_ids", None)
    if isinstance(configured, (set, frozenset, tuple, list)):
        return frozenset(str(item) for item in configured if str(item).strip())
    try:
        return frozenset(get_physical_graph_node_ids(graph))
    except Exception:
        return frozenset()


def _node_stream_mode(node_id: object) -> str:
    metadata = get_node_runtime_metadata(str(node_id or "").strip())
    return metadata.stream_mode if metadata is not None else "none"


def _state_values(state_snapshot) -> dict:
    values = getattr(state_snapshot, "values", None)
    return values if isinstance(values, dict) else {}


def _pending_interrupt_values(state_snapshot) -> list[dict]:
    pending: list[dict] = []
    for task in getattr(state_snapshot, "tasks", ()) or ():
        for interrupt_item in getattr(task, "interrupts", ()) or ():
            value = getattr(interrupt_item, "value", None)
            pending.append(
                value
                if isinstance(value, dict)
                else {"type": "plan_review", "value": value}
            )
    return pending


def _pending_interrupt_type(state_snapshot) -> str:
    for value in _pending_interrupt_values(state_snapshot):
        value_type = str(value.get("type") or "")
        if value_type:
            return value_type
    return ""


def _safe_profile_completion_request(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    raw_request = value.get("profile_completion_request")
    if not isinstance(raw_request, dict):
        raw_request = value
    title = sanitize_error_message(raw_request.get("title", ""), max_chars=160)
    fields: list[dict] = []
    for raw_field in raw_request.get("fields") or []:
        if not isinstance(raw_field, dict):
            continue
        key = sanitize_error_message(raw_field.get("key", ""), max_chars=80)
        label = sanitize_error_message(raw_field.get("label", ""), max_chars=120)
        if not key or not label:
            continue
        max_chars = _safe_int(raw_field.get("max_chars"), default=400)
        fields.append(
            {
                "key": key,
                "label": label,
                "required": raw_field.get("required") is True,
                "max_chars": max(1, min(max_chars, 1000)),
            }
        )
        if len(fields) >= 12:
            break
    return {"title": title, "fields": fields} if title or fields else {}


def _complete_profile_completion_request(value: object) -> dict:
    request = _safe_profile_completion_request(value)
    fields = request.get("fields")
    if not request.get("title"):
        return {}
    if not isinstance(fields, list) or not fields:
        return {}
    return request


def _profile_completion_request_from_trace_event(event: object) -> dict:
    if not isinstance(event, dict):
        return {}
    direct_request = _complete_profile_completion_request(
        event.get("profile_completion_request")
    )
    if direct_request:
        return direct_request
    return _complete_profile_completion_request(event)


def _pending_profile_completion_request(
    state_snapshot, values: dict | None = None
) -> dict:
    for interrupt_value in _pending_interrupt_values(state_snapshot):
        if interrupt_value.get("type") == "profile_completion_required":
            return _safe_profile_completion_request(interrupt_value)
    saved = (values or {}).get("profile_completion_request")
    return _safe_profile_completion_request(saved) if isinstance(saved, dict) else {}


def _has_checkpoint_state(state_snapshot) -> bool:
    values = _state_values(state_snapshot)
    if values:
        return True
    if getattr(state_snapshot, "next", None):
        return True
    return bool(_pending_interrupt_values(state_snapshot))


def _missing_run_control_fields(values: dict) -> list[str]:
    return [field for field in RUN_CONTROL_FIELDS if field not in values]


def _context_window_status(values: dict) -> tuple[dict, dict]:
    request_window = values.get("request_context_window")
    request_context_window = {
        "current_request_id": "",
        "current_node": "",
        "last_event_count": 0,
    }
    if isinstance(request_window, dict):
        request_context_window.update(
            {
                "current_request_id": sanitize_error_message(
                    request_window.get("current_request_id", ""),
                    max_chars=120,
                ),
                "current_node": sanitize_error_message(
                    request_window.get("current_node", ""),
                    max_chars=120,
                ),
                "last_event_count": _safe_int(
                    request_window.get("last_event_count"),
                    default=0,
                ),
            }
        )
    raw_usage_history = values.get("context_usage_history")
    usage_history: list = (
        raw_usage_history if isinstance(raw_usage_history, list) else []
    )
    resource_artifacts_by_type = values.get("resource_artifacts_by_type")
    last_generated_artifacts = values.get("last_generated_artifacts")
    last_resource_payload_value = values.get("last_resource_final_payload")
    last_resource_payload: dict = (
        last_resource_payload_value
        if isinstance(last_resource_payload_value, dict)
        else {}
    )
    last_qa_response_value = values.get("last_qa_response")
    last_qa_response: dict = (
        last_qa_response_value if isinstance(last_qa_response_value, dict) else {}
    )
    workspace_status = workspace_status_payload(values.get("task_workspace"))
    raw_manifest_history = values.get("llm_input_manifests")
    manifest_history: list = (
        raw_manifest_history if isinstance(raw_manifest_history, list) else []
    )
    background_window = (
        values.get("background_context_window")
        if isinstance(values.get("background_context_window"), dict)
        else {}
    )
    background_status = background_context_status_payload(background_window)
    influence_status = influence_status_payload(
        values.get("context_influence_ledger")
        if isinstance(values.get("context_influence_ledger"), dict)
        else {}
    )
    raw_usage_reports = values.get("context_usage_reports")
    usage_reports = raw_usage_reports if isinstance(raw_usage_reports, list) else []
    raw_activity_timeline = values.get("activity_timeline")
    activity_status = activity_timeline_status(
        raw_activity_timeline if isinstance(raw_activity_timeline, list) else []
    )
    thread_context_window = {
        "context_usage_history_count": len(usage_history),
        "artifact_count": _artifact_count(
            resource_artifacts_by_type,
            last_generated_artifacts,
        ),
        "conversation_summary_present": bool(
            str(values.get("conversation_summary") or "").strip()
        ),
        "last_context_policy_by_node_keys": _dict_keys(
            values.get("last_context_policy_by_node")
        ),
        "last_provider_supply_by_node_keys": _dict_keys(
            values.get("last_provider_supply_by_node")
        ),
        "last_context_selection_by_node_keys": _dict_keys(
            values.get("last_context_selection_by_node")
        ),
        "last_context_applied_by_node_keys": _dict_keys(
            values.get("last_context_applied_by_node")
        ),
        "last_resource_subnodes_count": len(values.get("last_resource_subnodes") or [])
        if isinstance(values.get("last_resource_subnodes"), list)
        else 0,
        "last_resource_final_payload_present": bool(last_resource_payload),
        "last_resource_final_resource_type": str(
            last_resource_payload.get("resource_type") or ""
        ),
        "last_qa_response_present": bool(last_qa_response),
        "last_qa_scope": str(last_qa_response.get("qa_scope") or ""),
        "background_context_window": background_window,
        "context_influence_ledger": influence_status,
        "context_influence_entry_count": influence_status.get("entry_count", 0),
        "context_influence_total_recorded": influence_status.get("total_recorded", 0),
        **background_status,
        "llm_input_manifest_count": len(manifest_history),
        "context_usage_report_count": len(usage_reports),
        "context_usage_report_present": bool(_last_context_usage_report(values)),
        "graph_version": sanitize_error_message(
            values.get("graph_version", ""),
            max_chars=180,
        ),
        **activity_status,
        **workspace_status,
    }
    return request_context_window, thread_context_window


def _last_llm_input_manifest(values: dict) -> dict:
    manifest = values.get("llm_input_manifest")
    if isinstance(manifest, dict) and manifest:
        return manifest
    history = values.get("llm_input_manifests")
    if isinstance(history, list) and history:
        latest = history[0]
        if isinstance(latest, dict):
            return latest
    return {}


def _last_context_usage_report(values: dict) -> dict:
    report = values.get("context_usage_report")
    if isinstance(report, dict) and report:
        return report
    history = values.get("context_usage_reports")
    if isinstance(history, list) and history:
        latest = history[0]
        if isinstance(latest, dict):
            return latest
    return {}


def _last_recommendation_final_payload(
    values: dict,
    *,
    thread_id: str,
) -> RecommendationFinalV1 | None:
    raw = values.get("last_recommendation_final_payload")
    if raw in ({}, None):
        return None
    final = validate_recommendation_final_v1(raw)
    if final.thread_id != thread_id:
        raise ValueError("recommendation final thread_id does not match thread status")
    return final


def _activity_timeline(values: dict) -> list[dict]:
    timeline = values.get("activity_timeline")
    return merge_activity_timeline([], timeline if isinstance(timeline, list) else [])


def _dict_keys(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(sanitize_error_message(key, max_chars=120) for key in value)


def _artifact_count(by_type: object, generated: object) -> int:
    total = len(by_type) if isinstance(by_type, dict) else 0
    total += len(generated) if isinstance(generated, list) else 0
    return total


def _session_context_memory_ledger(
    values: dict,
    *,
    thread_id: str,
) -> SessionContextMemoryLedgerV1:
    raw = values.get("session_context_memory_ledger")
    if isinstance(raw, dict) and raw:
        ledger = SessionContextMemoryLedgerV1.model_validate(raw)
        if ledger.thread_id != thread_id:
            raise ValueError("session context memory ledger thread_id mismatch")
        return ledger
    return new_session_context_memory_ledger(thread_id)


def _thread_context_window_v3(
    values: dict,
    *,
    thread_id: str,
    updating: bool,
) -> dict:
    ledger = _session_context_memory_ledger(values, thread_id=thread_id)
    return build_thread_context_window_v3(
        ledger,
        updating=updating,
    ).model_dump(mode="json")


def _active_session_context_fields(values: dict, *, thread_id: str) -> dict:
    ledger = _session_context_memory_ledger(values, thread_id=thread_id)
    return {
        "session_context_memory_ledger": ledger.model_dump(mode="json"),
        "thread_context_window_v3": build_thread_context_window_v3(
            ledger,
            updating=True,
        ).model_dump(mode="json"),
    }


def _new_request_status_values(snapshot_values: dict, initial_run_values: dict) -> dict:
    """Combine current run control with persistent thread context for status UI."""
    values = dict(snapshot_values or {})
    previous_history = values.get("context_usage_history")
    previous_manifests = values.get("llm_input_manifests")
    previous_manifest = values.get("llm_input_manifest")
    previous_ledger = values.get("thread_context_ledger")
    previous_background = values.get("background_context_window")
    previous_resource_payload = values.get("last_resource_final_payload")
    previous_recommendation_payload = values.get("last_recommendation_final_payload")
    previous_usage_report = values.get("context_usage_report")
    previous_usage_reports = values.get("context_usage_reports")
    previous_activity_timeline = values.get("activity_timeline")
    previous_session_memory_ledger = values.get("session_context_memory_ledger")
    previous_thread_window_v3 = values.get("thread_context_window_v3")
    values.update(initial_run_values or {})
    if isinstance(previous_history, list):
        values["context_usage_history"] = previous_history
    if isinstance(previous_manifests, list):
        values["llm_input_manifests"] = previous_manifests
    if isinstance(previous_manifest, dict):
        values["llm_input_manifest"] = previous_manifest
    if isinstance(previous_ledger, dict):
        values["thread_context_ledger"] = previous_ledger
    if isinstance(previous_background, dict):
        values["background_context_window"] = previous_background
    if isinstance(previous_resource_payload, dict):
        values["last_resource_final_payload"] = previous_resource_payload
    if isinstance(previous_recommendation_payload, dict):
        values["last_recommendation_final_payload"] = previous_recommendation_payload
    if isinstance(previous_usage_report, dict):
        values["context_usage_report"] = previous_usage_report
    if isinstance(previous_usage_reports, list):
        values["context_usage_reports"] = previous_usage_reports
    if isinstance(previous_activity_timeline, list):
        values["activity_timeline"] = previous_activity_timeline
    if isinstance(previous_session_memory_ledger, dict):
        values["session_context_memory_ledger"] = previous_session_memory_ledger
    if isinstance(previous_thread_window_v3, dict):
        values["thread_context_window_v3"] = previous_thread_window_v3
    return values


def _thread_status_from_snapshot(
    thread_id: str, state_snapshot
) -> ThreadStatusResponse:
    values = _state_values(state_snapshot)
    pending_type = _pending_interrupt_type(state_snapshot)
    profile_completion_request = _pending_profile_completion_request(
        state_snapshot,
        values,
    )
    missing_fields = _missing_run_control_fields(values)
    request_context_window, thread_context_window = _context_window_status(values)
    thread_context_window_v3 = _thread_context_window_v3(
        values,
        thread_id=thread_id,
        updating=False,
    )
    if missing_fields:
        return ThreadStatusResponse(
            thread_id=thread_id,
            schema_version="legacy",
            run_status=RUN_STATUS_IDLE,
            resume_available=pending_type == "profile_completion_required",
            pending_interrupt_type=pending_type,
            current_node=str(values.get("current_node") or ""),
            last_completed_node=str(values.get("last_completed_node") or ""),
            context_usage=values.get("context_usage")
            if isinstance(values.get("context_usage"), dict)
            else {},
            context_usage_history=values.get("context_usage_history")
            if isinstance(values.get("context_usage_history"), list)
            else [],
            context_usage_report=_last_context_usage_report(values),
            context_usage_report_count=len(values.get("context_usage_reports") or [])
            if isinstance(values.get("context_usage_reports"), list)
            else 0,
            activity_timeline=_activity_timeline(values),
            activity_timeline_count=len(_activity_timeline(values)),
            graph_version=str(values.get("graph_version") or ""),
            last_llm_input_manifest=_last_llm_input_manifest(values),
            llm_input_manifest_count=len(values.get("llm_input_manifests") or [])
            if isinstance(values.get("llm_input_manifests"), list)
            else 0,
            background_context_window=values.get("background_context_window")
            if isinstance(values.get("background_context_window"), dict)
            else {},
            context_influence_ledger=thread_context_window.get(
                "context_influence_ledger", {}
            ),
            last_resource_final_payload=values.get("last_resource_final_payload")
            if isinstance(values.get("last_resource_final_payload"), dict)
            else {},
            last_recommendation_final_payload=_last_recommendation_final_payload(
                values,
                thread_id=thread_id,
            ),
            last_qa_response=values.get("last_qa_response")
            if isinstance(values.get("last_qa_response"), dict)
            else {},
            request_context_window=request_context_window,
            thread_context_window=thread_context_window,
            thread_context_window_v3=thread_context_window_v3,
            profile_completion_request=profile_completion_request,
            missing_run_control_fields=missing_fields,
            message="legacy checkpoint does not include run-control fields",
        )
    raw_run_status = str(values.get("run_status") or RUN_STATUS_UNKNOWN)
    terminal_statuses = {RUN_STATUS_COMPLETED, RUN_STATUS_ERROR, RUN_STATUS_STOPPED}
    active_statuses = {RUN_STATUS_RUNNING, RUN_STATUS_STOPPING, RUN_STATUS_CONTINUING}
    if raw_run_status in terminal_statuses:
        run_status = raw_run_status
    elif pending_type:
        run_status = RUN_STATUS_STOPPED
    elif raw_run_status in active_statuses:
        run_status = RUN_STATUS_IDLE
    else:
        run_status = RUN_STATUS_IDLE

    return ThreadStatusResponse(
        thread_id=thread_id,
        schema_version=RUN_CONTROL_SCHEMA_VERSION,
        run_status=run_status,
        resume_available=pending_type in {"user_stop", "profile_completion_required"},
        pending_interrupt_type=pending_type
        or str(values.get("pending_interrupt_type") or ""),
        current_node=str(values.get("current_node") or ""),
        last_completed_node=str(values.get("last_completed_node") or ""),
        stopped_at=str(values.get("stopped_at") or ""),
        stop_reason=str(values.get("stop_reason") or ""),
        context_usage=values.get("context_usage")
        if isinstance(values.get("context_usage"), dict)
        else {},
        context_usage_history=values.get("context_usage_history")
        if isinstance(values.get("context_usage_history"), list)
        else [],
        context_usage_report=_last_context_usage_report(values),
        context_usage_report_count=len(values.get("context_usage_reports") or [])
        if isinstance(values.get("context_usage_reports"), list)
        else 0,
        activity_timeline=_activity_timeline(values),
        activity_timeline_count=len(_activity_timeline(values)),
        graph_version=str(values.get("graph_version") or ""),
        last_llm_input_manifest=_last_llm_input_manifest(values),
        llm_input_manifest_count=len(values.get("llm_input_manifests") or [])
        if isinstance(values.get("llm_input_manifests"), list)
        else 0,
        background_context_window=values.get("background_context_window")
        if isinstance(values.get("background_context_window"), dict)
        else {},
        context_influence_ledger=thread_context_window.get(
            "context_influence_ledger", {}
        ),
        last_resource_final_payload=values.get("last_resource_final_payload")
        if isinstance(values.get("last_resource_final_payload"), dict)
        else {},
        last_recommendation_final_payload=_last_recommendation_final_payload(
            values,
            thread_id=thread_id,
        ),
        last_qa_response=values.get("last_qa_response")
        if isinstance(values.get("last_qa_response"), dict)
        else {},
        request_context_window=request_context_window,
        thread_context_window=thread_context_window,
        thread_context_window_v3=thread_context_window_v3,
        profile_completion_request=profile_completion_request,
        missing_run_control_fields=[],
    )


def _thread_status_from_active_run(
    thread_id: str, active_run: dict
) -> ThreadStatusResponse:
    request_context_window = active_run.get("request_context_window")
    thread_context_window = active_run.get("thread_context_window")
    profile_completion_request = active_run.get("profile_completion_request")
    active_window_v3 = active_run.get("thread_context_window_v3")
    thread_context_window_v3 = (
        active_window_v3
        if isinstance(active_window_v3, dict) and active_window_v3
        else _thread_context_window_v3(
            active_run,
            thread_id=thread_id,
            updating=True,
        )
    )
    return ThreadStatusResponse(
        thread_id=thread_id,
        schema_version=RUN_CONTROL_SCHEMA_VERSION,
        run_status=str(active_run.get("run_status") or RUN_STATUS_RUNNING),
        resume_available=bool(active_run.get("resume_available", False)),
        pending_interrupt_type=str(active_run.get("pending_interrupt_type") or ""),
        current_node=str(active_run.get("current_node") or ""),
        last_completed_node=str(active_run.get("last_completed_node") or ""),
        stopped_at=str(active_run.get("stopped_at") or ""),
        stop_reason=str(active_run.get("stop_reason") or ""),
        context_usage=active_run.get("context_usage")
        if isinstance(active_run.get("context_usage"), dict)
        else {},
        context_usage_history=active_run.get("context_usage_history")
        if isinstance(active_run.get("context_usage_history"), list)
        else [],
        context_usage_report=_last_context_usage_report(active_run),
        context_usage_report_count=len(active_run.get("context_usage_reports") or [])
        if isinstance(active_run.get("context_usage_reports"), list)
        else 0,
        activity_timeline=_activity_timeline(active_run),
        activity_timeline_count=len(_activity_timeline(active_run)),
        graph_version=str(active_run.get("graph_version") or ""),
        last_llm_input_manifest=active_run.get("llm_input_manifest")
        if isinstance(active_run.get("llm_input_manifest"), dict)
        else {},
        llm_input_manifest_count=len(active_run.get("llm_input_manifests") or [])
        if isinstance(active_run.get("llm_input_manifests"), list)
        else 0,
        background_context_window=active_run.get("background_context_window")
        if isinstance(active_run.get("background_context_window"), dict)
        else {},
        context_influence_ledger=active_run.get("context_influence_ledger")
        if isinstance(active_run.get("context_influence_ledger"), dict)
        else (
            thread_context_window.get("context_influence_ledger", {})
            if isinstance(thread_context_window, dict)
            else {}
        ),
        last_resource_final_payload=active_run.get("last_resource_final_payload")
        if isinstance(active_run.get("last_resource_final_payload"), dict)
        else {},
        last_recommendation_final_payload=_last_recommendation_final_payload(
            active_run,
            thread_id=thread_id,
        ),
        request_context_window=request_context_window
        if isinstance(request_context_window, dict)
        else {"current_request_id": "", "current_node": "", "last_event_count": 0},
        thread_context_window=thread_context_window
        if isinstance(thread_context_window, dict)
        else {
            "context_usage_history_count": 0,
            "artifact_count": 0,
            "conversation_summary_present": False,
            "last_context_policy_by_node_keys": [],
            "last_provider_supply_by_node_keys": [],
            "last_context_selection_by_node_keys": [],
            "last_context_applied_by_node_keys": [],
            "last_resource_subnodes_count": 0,
            "llm_input_manifest_count": 0,
            "background_context_window": {},
            "background_context_window_present": False,
            "background_context_window_used_tokens": 0,
            "background_context_window_max_tokens": 0,
            "background_context_window_used_ratio": 0.0,
            "background_context_window_updated_at": "",
            "context_influence_ledger": {},
            "context_influence_entry_count": 0,
            "context_influence_total_recorded": 0,
            "workspace_present": False,
            "workspace_active_subject": "",
            "workspace_evidence_summary_count": 0,
            "workspace_gap_count": 0,
            "workspace_artifact_count": 0,
            "workspace_updated_at": "",
        },
        thread_context_window_v3=thread_context_window_v3,
        profile_completion_request=profile_completion_request
        if isinstance(profile_completion_request, dict)
        else {},
        missing_run_control_fields=[],
    )


def _valid_state_update_node(
    values: dict,
    state: dict | None = None,
    *,
    runtime_node_ids: frozenset[str] = frozenset(),
) -> str:
    candidates = (
        values.get("current_node"),
        (state or {}).get("current_node"),
        (state or {}).get("last_completed_node"),
        "supervisor",
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text in runtime_node_ids:
            return text
    if "supervisor" in runtime_node_ids or not runtime_node_ids:
        return "supervisor"
    raise RuntimeError("compiled graph does not expose a valid state update node")


async def safe_update_thread_state(
    graph,
    config: dict,
    values: dict,
    *,
    state: dict | None = None,
    as_node: str | None = None,
) -> None:
    """Update checkpoint state using a real graph node as LangGraph writer."""
    runtime_node_ids = _runtime_graph_node_ids(graph)
    node = str(as_node or "").strip()
    if node not in runtime_node_ids:
        node = _valid_state_update_node(
            values,
            state,
            runtime_node_ids=runtime_node_ids,
        )
    await graph.aupdate_state(config, values, as_node=node)


async def _update_run_state(
    graph,
    config: dict,
    values: dict,
    *,
    state: dict | None = None,
    as_node: str | None = None,
) -> None:
    await safe_update_thread_state(
        graph,
        config,
        values,
        state=state,
        as_node=as_node,
    )


async def _try_update_run_state(
    graph,
    config: dict,
    values: dict,
    *,
    state: dict | None = None,
    persist_checkpoint: bool = True,
    update_active_on_failure: bool = True,
) -> bool:
    thread_id = _thread_id_from_update(config=config, values=values, state=state)
    if not persist_checkpoint:
        if thread_id and get_active_run(thread_id) is not None:
            update_active_run(thread_id, values)
        return True
    try:
        await _update_run_state(graph, config, values, state=state)
        if thread_id and get_active_run(thread_id) is not None:
            update_active_run(thread_id, values)
        return True
    except Exception as exc:
        if (
            update_active_on_failure
            and thread_id
            and get_active_run(thread_id) is not None
        ):
            update_active_run(thread_id, values)
        emit_a3_trace(
            logger,
            "run_state_update_failed",
            {
                "keys": sorted(values.keys()),
                "error_type": type(exc).__name__,
            },
            state=state or {},
            env_flag="LOG_A3_TRACE",
        )
        return False


def _thread_id_from_update(
    *,
    config: dict,
    values: dict,
    state: dict | None,
) -> str:
    configurable = config.get("configurable") if isinstance(config, dict) else {}
    candidates = (
        values.get("thread_id"),
        values.get("session_id"),
        (state or {}).get("thread_id"),
        (state or {}).get("session_id"),
        (configurable or {}).get("thread_id") if isinstance(configurable, dict) else "",
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


async def _update_context_window_state_from_trace(
    graph,
    config: dict,
    *,
    thread_id: str,
    request_context_events: list[dict],
    context_usage_history: list[dict],
    last_context_policy_by_node: dict[str, dict],
    last_provider_supply_by_node: dict[str, dict],
    last_context_selection_by_node: dict[str, dict],
    last_context_applied_by_node: dict[str, dict],
    last_drop_reasons_by_node: dict[str, dict],
    last_resource_subnodes: list[dict],
    current_node: str,
) -> None:
    current_request_id = (
        str(request_context_events[-1].get("request_id", "") or "")
        if request_context_events
        else ""
    )
    values = {
        "request_context_window": {
            "current_request_id": current_request_id,
            "current_node": current_node,
            "last_event_count": len(request_context_events),
        },
        "context_window_events": list(request_context_events),
        "last_context_policy_by_node": dict(last_context_policy_by_node),
        "last_provider_supply_by_node": dict(last_provider_supply_by_node),
        "last_context_selection_by_node": dict(last_context_selection_by_node),
        "last_context_applied_by_node": dict(last_context_applied_by_node),
        "last_drop_reasons_by_node": dict(last_drop_reasons_by_node),
        "last_resource_subnodes": list(last_resource_subnodes),
    }
    active_run = get_active_run(thread_id)
    active_thread_window = (
        dict(active_run.get("thread_context_window") or {})
        if isinstance(active_run, dict)
        else {}
    )
    active_thread_window.update(
        {
            "context_usage_history_count": len(context_usage_history),
            "context_usage_history_kind": "llm_call_history",
            "last_context_policy_by_node_keys": sorted(last_context_policy_by_node),
            "last_provider_supply_by_node_keys": sorted(last_provider_supply_by_node),
            "last_context_selection_by_node_keys": sorted(
                last_context_selection_by_node
            ),
            "last_context_applied_by_node_keys": sorted(last_context_applied_by_node),
            "last_resource_subnodes_count": len(last_resource_subnodes),
        }
    )
    values["thread_context_window"] = active_thread_window
    updated = await _try_update_run_state(
        graph,
        config,
        values,
        state={
            "request_id": current_request_id,
            "thread_id": thread_id,
            "session_id": thread_id,
            "current_node": current_node,
        },
        persist_checkpoint=False,
    )
    emit_a3_trace(
        logger,
        "context_window_state_updated"
        if updated
        else "context_window_state_update_failed",
        {
            "request_id": current_request_id,
            "current_node": current_node,
            "request_event_count": len(request_context_events),
            "context_usage_history_count": len(context_usage_history),
            "context_usage_history_kind": "llm_call_history",
            "policy_node_count": len(last_context_policy_by_node),
            "supply_node_count": len(last_provider_supply_by_node),
            "selection_node_count": len(last_context_selection_by_node),
            "applied_node_count": len(last_context_applied_by_node),
            "resource_subnode_count": len(last_resource_subnodes),
        },
        state={
            "request_id": current_request_id,
            "thread_id": thread_id,
            "session_id": thread_id,
        },
        env_flag="LOG_A3_TRACE",
    )


async def _update_llm_manifest_state_from_trace(
    graph,
    config: dict,
    *,
    thread_id: str,
    event: dict,
    llm_input_manifests: list[dict],
    state_context: dict,
) -> tuple[dict, dict, dict, list[dict]]:
    payload = llm_input_manifest_trace_payload(event)
    llm_input_manifests[:] = merge_llm_input_manifest_history(
        llm_input_manifests,
        [payload],
    )
    active_run = get_active_run(thread_id)
    existing_ledger = (
        active_run.get("thread_context_ledger")
        if isinstance(active_run, dict)
        and isinstance(active_run.get("thread_context_ledger"), dict)
        else state_context.get("thread_context_ledger")
        if isinstance(state_context.get("thread_context_ledger"), dict)
        else {}
    )
    ledger_update = build_thread_context_ledger_update(
        existing=existing_ledger,
        manifest=payload,
    )
    background_window = build_background_context_window(
        manifest=payload,
        state=state_context,
        manifest_count=len(llm_input_manifests),
    )
    active_thread_window = (
        dict(active_run.get("thread_context_window") or {})
        if isinstance(active_run, dict)
        else {}
    )
    active_thread_window.update(
        {
            "llm_input_manifest_count": len(llm_input_manifests),
            "background_context_window": background_window,
            **background_context_status_payload(background_window),
        }
    )
    await _try_update_run_state(
        graph,
        config,
        {
            "llm_input_manifest": payload,
            "llm_input_manifests": [payload],
            "thread_context_ledger": ledger_update,
            "background_context_window": background_window,
            "thread_context_window": active_thread_window,
        },
        state={"thread_id": thread_id, "session_id": thread_id},
        persist_checkpoint=True,
    )
    return (
        dict(payload),
        dict(ledger_update),
        dict(background_window),
        llm_input_manifests,
    )


async def _update_session_context_memory_from_trace(
    graph,
    config: dict,
    *,
    thread_id: str,
    event: dict,
    state_context: dict,
) -> dict:
    """Persist one actual provider-dispatch item and its V3 projection atomically."""

    record = ContextInjectionRecordV1.model_validate(
        {
            key: event.get(key)
            for key in (
                "schema_version",
                "record_id",
                "dispatch_id",
                "request_id",
                "call_id",
                "attempt",
                "manifest_id",
                "thread_id",
                "item",
                "dispatched_at",
            )
        }
    )
    if record.thread_id != thread_id:
        raise ValueError("context injection trace thread_id mismatch")
    active_run = get_active_run(thread_id)
    ledger_values = (
        active_run
        if isinstance(active_run, dict)
        and isinstance(active_run.get("session_context_memory_ledger"), dict)
        else state_context
    )
    ledger = _session_context_memory_ledger(ledger_values, thread_id=thread_id)
    updated_ledger = record_context_injection(ledger, record)
    window_v3 = build_thread_context_window_v3(
        updated_ledger,
        updating=False,
    ).model_dump(mode="json")
    state_update = {
        "session_context_memory_ledger": updated_ledger.model_dump(mode="json"),
        "thread_context_window_v3": window_v3,
    }
    await safe_update_thread_state(
        graph,
        config,
        state_update,
        state={
            "request_id": record.request_id,
            "thread_id": thread_id,
            "session_id": thread_id,
        },
    )
    if isinstance(active_run, dict):
        update_active_run(thread_id, state_update)
    state_context.update(state_update)
    return window_v3


async def _update_last_provider_dispatch_from_trace(
    graph,
    config: dict,
    *,
    thread_id: str,
    event: dict,
    state_context: dict,
) -> ProviderBoundUsageV1:
    """Persist the latest actual business-provider input for future compaction."""

    usage = provider_bound_usage_from_trace(event)
    if usage.thread_id != thread_id:
        raise ValueError("provider dispatch trace thread_id mismatch")
    payload = usage.model_dump(mode="json")
    await safe_update_thread_state(
        graph,
        config,
        {"last_provider_dispatch": payload},
        state={
            "request_id": usage.request_id,
            "thread_id": thread_id,
            "session_id": thread_id,
        },
    )
    active_run = get_active_run(thread_id)
    if isinstance(active_run, dict):
        update_active_run(thread_id, {"last_provider_dispatch": payload})
    state_context["last_provider_dispatch"] = payload
    return usage


async def _prepare_full_compaction_for_new_request(
    graph,
    config: dict,
    *,
    thread_id: str,
    request_id: str,
    snapshot_values: dict,
    state_input: dict,
) -> tuple[dict, dict | None]:
    """Commit a validated full compaction before starting a new graph run."""

    compaction_config = get_full_compaction_config()
    decision = evaluate_full_compaction(
        snapshot_values.get("last_provider_dispatch"),
        config=compaction_config,
    )
    emit_a3_trace(
        logger,
        "full_compaction.decision",
        decision.model_dump(mode="json"),
        state={
            "request_id": request_id,
            "thread_id": thread_id,
            "session_id": thread_id,
        },
        env_flag="LOG_A3_TRACE",
    )
    if not decision.eligible:
        return dict(snapshot_values), None

    history_messages = snapshot_values.get("messages")
    if not isinstance(history_messages, list):
        history_messages = []
    request_messages = state_input.get("messages")
    if not isinstance(request_messages, list):
        raise ValueError("new request messages are unavailable for compaction")
    model_history = [*history_messages, *request_messages]
    boundary = build_compact_boundary(
        model_history,
        thread_id=thread_id,
        request_id=request_id,
        trigger_dispatch_id=decision.dispatch_id,
        retain_recent_rounds=compaction_config.retain_recent_rounds,
    )
    if boundary is None:
        emit_a3_trace(
            logger,
            "full_compaction.skipped",
            {
                "reason": "insufficient_history",
                "dispatch_id": decision.dispatch_id,
                "retained_recent_rounds": compaction_config.retain_recent_rounds,
            },
            state={
                "request_id": request_id,
                "thread_id": thread_id,
                "session_id": thread_id,
            },
            env_flag="LOG_A3_TRACE",
        )
        return dict(snapshot_values), None

    summary = await invoke_conversation_compaction(
        boundary=boundary,
        messages=model_history,
        state=snapshot_values,
        config=compaction_config,
    )
    before_projection = build_model_view_projection(
        model_history,
        state=snapshot_values,
    )
    candidate_state = {
        **snapshot_values,
        "thread_id": thread_id,
        "session_id": thread_id,
        "compact_boundary": boundary.model_dump(mode="json"),
        "conversation_summary_v2": summary.model_dump(mode="json"),
        "conversation_summary": summary.summary,
    }
    after_projection = build_model_view_projection(
        model_history,
        state=candidate_state,
    )

    ledger = _session_context_memory_ledger(
        snapshot_values,
        thread_id=thread_id,
    )
    retained_item_ids = list(ledger.active_items)
    compacted_at = datetime.now(timezone.utc)
    updated_ledger = apply_context_memory_compaction(
        ledger,
        boundary_id=boundary.boundary_id,
        retained_logical_item_ids=retained_item_ids,
        compacted_at=compacted_at,
        before_tokens=ledger.retained_memory_tokens,
        after_tokens=ledger.retained_memory_tokens,
    )
    result = CompactionResultV1(
        status="compacted",
        boundary_id=boundary.boundary_id,
        trigger_dispatch_id=decision.dispatch_id,
        compacted_at=compacted_at,
        trigger_input_tokens=decision.observed_input_tokens,
        context_window_limit_tokens=decision.context_window_limit_tokens,
        trigger_ratio=decision.observed_ratio,
        compact_ratio=decision.compact_ratio,
        model_view_before_tokens=(before_projection.projection.output_estimated_tokens),
        model_view_after_tokens=after_projection.projection.output_estimated_tokens,
        compacted_message_count=len(boundary.compacted_messages),
        retained_message_count=boundary.retained_message_count,
        summary_fingerprint=summary_fingerprint(summary),
        ledger_before_tokens=ledger.retained_memory_tokens,
        ledger_after_tokens=updated_ledger.retained_memory_tokens,
    )
    window_v3 = build_thread_context_window_v3(
        updated_ledger,
        updating=False,
    ).model_dump(mode="json")
    state_update = {
        "conversation_summary": summary.summary,
        "conversation_summary_v2": summary.model_dump(mode="json"),
        "compact_boundary": boundary.model_dump(mode="json"),
        "compaction_result": result.model_dump(mode="json"),
        "session_context_memory_ledger": updated_ledger.model_dump(mode="json"),
        "thread_context_window_v3": window_v3,
    }
    await safe_update_thread_state(
        graph,
        config,
        state_update,
        state={
            "request_id": request_id,
            "thread_id": thread_id,
            "session_id": thread_id,
        },
        as_node="supervisor",
    )
    state_input.update(state_update)
    updated_values = {**snapshot_values, **state_update}
    emit_a3_trace(
        logger,
        "full_compaction.committed",
        result.model_dump(mode="json"),
        state={
            "request_id": request_id,
            "thread_id": thread_id,
            "session_id": thread_id,
        },
        env_flag="LOG_A3_TRACE",
    )
    return updated_values, result.model_dump(mode="json")


def _resource_final_payload(final_state: dict) -> dict | None:
    """Return the authoritative Resource Final V3 state, if present."""

    raw_v3 = final_state.get("resource_final_v3")
    if "resource_final_v3" not in final_state or raw_v3 == {}:
        return None
    resource_final = validate_resource_final_v3(raw_v3)
    thread_id = str(final_state.get("thread_id") or "").strip()
    request_id = str(final_state.get("request_id") or "").strip()
    if thread_id and resource_final.thread_id != thread_id:
        raise ValueError("Resource Final V3 thread_id does not match runtime state")
    if request_id and resource_final.request_id != request_id:
        raise ValueError("Resource Final V3 request_id does not match runtime state")
    return resource_final.model_dump(mode="json")


def _recommendation_final_payload(final_state: dict) -> dict | None:
    """Return the current request's authoritative Recommendation Final V1."""

    raw = final_state.get("recommendation_final_v1")
    if "recommendation_final_v1" not in final_state or raw == {}:
        return None
    recommendation_final = validate_recommendation_final_v1(raw)
    thread_id = str(final_state.get("thread_id") or "").strip()
    request_id = str(final_state.get("request_id") or "").strip()
    if not thread_id or not request_id:
        raise ValueError(
            "Recommendation Final V1 requires runtime thread_id and request_id"
        )
    if recommendation_final.thread_id != thread_id:
        raise ValueError(
            "Recommendation Final V1 thread_id does not match runtime state"
        )
    if recommendation_final.request_id != request_id:
        raise ValueError(
            "Recommendation Final V1 request_id does not match runtime state"
        )
    return recommendation_final.model_dump(mode="json")


def _dev_memory_clear_enabled() -> bool:
    """Return whether the dev-only persistent-memory clear endpoint is enabled."""
    env_values = {
        (os.getenv("APP_ENV") or "").strip().lower(),
        (os.getenv("A3_ENV") or "").strip().lower(),
    }
    if env_values & {"production", "prod"}:
        return False
    return bool(get_setting("development.enable_dev_memory_clear", False))


async def clear_persistent_memory_for_thread(graph, thread_id: str) -> dict:
    """Clear persistent memory fields for a thread in development mode."""
    if not _dev_memory_clear_enabled():
        raise HTTPException(status_code=403, detail="Dev memory clear is disabled")

    config = make_thread_config(thread_id)
    cleared_fields = [
        "conversation_summary",
        "conversation_summary_v2",
        "compact_boundary",
        "compaction_result",
        "last_provider_dispatch",
        "evidence_summary_memory",
        "evidence_gap_memory",
        "episodic_memory_results",
        "semantic_memory_results",
        "task_workspace",
        "workspace_events",
        "resource_artifacts_by_type",
        "last_generated_artifacts",
        "last_resource_final_payload",
        "recommendation_final_v1",
        "last_recommendation_final_payload",
        "last_qa_response",
        "llm_input_manifest",
        "llm_input_manifests",
        "thread_context_ledger",
        "session_context_memory_ledger",
        "thread_context_window_v3",
        "background_context_window",
        "context_continuity",
        "context_influence_ledger",
        "context_usage_report",
        "context_usage_reports",
        "activity_timeline",
    ]
    values = {
        "conversation_summary": "",
        "conversation_summary_v2": {},
        "compact_boundary": {},
        "compaction_result": {},
        "last_provider_dispatch": {},
        "evidence_summary_memory": MEMORY_CLEAR,
        "evidence_gap_memory": MEMORY_CLEAR,
        "episodic_memory_results": [],
        "semantic_memory_results": [],
        "task_workspace": TASK_WORKSPACE_CLEAR,
        "workspace_events": WORKSPACE_EVENTS_CLEAR,
        "resource_artifacts_by_type": DICT_CLEAR,
        "last_generated_artifacts": GENERATED_ARTIFACTS_CLEAR,
        "last_resource_final_payload": DICT_CLEAR,
        "recommendation_final_v1": DICT_CLEAR,
        "last_recommendation_final_payload": DICT_CLEAR,
        "last_qa_response": {},
        "llm_input_manifest": {},
        "llm_input_manifests": LLM_INPUT_MANIFESTS_CLEAR,
        "thread_context_ledger": DICT_CLEAR,
        "session_context_memory_ledger": SESSION_CONTEXT_MEMORY_LEDGER_CLEAR,
        "thread_context_window_v3": {},
        "background_context_window": {},
        "context_continuity": {},
        "context_influence_ledger": CONTEXT_INFLUENCE_LEDGER_CLEAR,
        "context_usage_report": {},
        "context_usage_reports": CONTEXT_USAGE_REPORTS_CLEAR,
        "activity_timeline": ACTIVITY_TIMELINE_CLEAR,
    }
    await safe_update_thread_state(
        graph,
        config,
        values,
        state={"thread_id": thread_id, "session_id": thread_id},
        as_node="supervisor",
    )

    trace_state = {
        "thread_id": thread_id,
        "session_id": thread_id,
        "cleared_fields": cleared_fields,
    }
    emit_a3_trace(
        logger,
        "dev_memory_clear",
        {
            "thread_id": thread_id,
            "cleared_fields": cleared_fields,
            "success": True,
        },
        state=trace_state,
        env_flag="LOG_A3_TRACE",
    )
    return {"ok": True, "thread_id": thread_id, "cleared_fields": cleared_fields}


async def _stream_graph_event_drafts(
    graph,
    input_data,
    config: dict,
    thread_id: str,
    *,
    request_id: str,
    preserve_context_history: bool = False,
) -> AsyncGenerator[AgentStreamEventDraftV2, None]:
    """Project LangGraph execution directly into native agent_stream_v2 drafts."""
    node_start_times: dict[str, float] = {}
    active_nodes: list[str] = []
    trace_events: list[dict] = []
    trace_sink_token = set_trace_event_sink(trace_events)
    evidence_progress_events: deque[EvidenceProgressV1] = deque()
    evidence_progress_sink_token = set_evidence_progress_sink(
        evidence_progress_events.append
    )
    provisional_events: list[dict] = []
    provisional_sink_token = set_provisional_event_sink(provisional_events.append)
    context_usage_history: list[dict] = []
    context_usage_reports: list[dict] = []
    llm_input_manifests: list[dict] = []
    manifest_state_context: dict = {}
    request_context_events: list[dict] = []
    last_context_policy_by_node: dict[str, dict] = {}
    last_provider_supply_by_node: dict[str, dict] = {}
    last_context_selection_by_node: dict[str, dict] = {}
    last_context_applied_by_node: dict[str, dict] = {}
    last_drop_reasons_by_node: dict[str, dict] = {}
    last_resource_subnodes: list[dict] = []
    terminal_resource_output: dict | None = None
    worker_interrupted_seen = False
    recovered_profile_completion_request: dict = {}
    activity_enabled = getattr(graph, "_a3_activity_events_enabled", False) is True
    runtime_graph_nodes = _runtime_graph_node_ids(graph)
    activity_timeline: list[dict] = []
    activity_sequence = 0
    provisional_block_open = False
    node_content_blocks: dict[str, tuple[str, int]] = {}
    next_content_block_index = 1
    if preserve_context_history:
        try:
            existing_snapshot = await graph.aget_state(config)
            existing_values = _state_values(existing_snapshot)
            manifest_state_context = dict(existing_values)
            existing_history = existing_values.get("context_usage_history")
            if isinstance(existing_history, list):
                context_usage_history = trim_context_usage_history(existing_history)
            existing_manifests = existing_values.get("llm_input_manifests")
            if isinstance(existing_manifests, list):
                llm_input_manifests = merge_llm_input_manifest_history(
                    existing_manifests,
                    [],
                )
            existing_reports = existing_values.get("context_usage_reports")
            if isinstance(existing_reports, list):
                context_usage_reports = merge_context_usage_report_history(
                    existing_reports,
                    [],
                )
            existing_activities = existing_values.get("activity_timeline")
            if isinstance(existing_activities, list):
                activity_timeline = merge_activity_timeline(
                    existing_activities,
                    [],
                )
                activity_sequence = next_activity_sequence(activity_timeline) - 1
        except Exception as exc:
            emit_a3_trace(
                logger,
                "run_state_read_failed",
                {
                    "operation": "load_context_history",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                state={"thread_id": thread_id, "session_id": thread_id},
                env_flag="LOG_A3_TRACE",
            )

    async def _record_activity(
        activity: ActivityEvent,
        *,
        persist_checkpoint: bool = False,
    ) -> AgentStreamEventDraftV2:
        nonlocal activity_timeline, activity_sequence
        activity_sequence = max(activity_sequence, activity.sequence)
        activity_timeline = merge_activity_timeline(
            activity_timeline,
            [activity.model_dump(mode="json")],
        )
        active_run = get_active_run(thread_id)
        active_thread_window = (
            dict(active_run.get("thread_context_window") or {})
            if isinstance(active_run, dict)
            else {}
        )
        active_thread_window.update(activity_timeline_status(activity_timeline))
        update_active_run(
            thread_id,
            {
                "activity_timeline": activity_timeline,
                "thread_context_window": active_thread_window,
            },
        )
        if persist_checkpoint:
            await _try_update_run_state(
                graph,
                config,
                {
                    "activity_timeline": activity_timeline,
                    "thread_context_window": active_thread_window,
                },
                state={
                    "request_id": request_id,
                    "thread_id": thread_id,
                    "session_id": thread_id,
                    "current_node": activity.node,
                },
                persist_checkpoint=True,
            )
        return _activity_update_draft(
            "activity_event",
            activity.model_dump(mode="json"),
        )

    async def _activity_from_trace(
        event: dict,
    ) -> AgentStreamEventDraftV2 | None:
        if not activity_enabled:
            return None
        try:
            activity = activity_from_trace_event(
                event,
                thread_id=thread_id,
                request_id=request_id,
                sequence=activity_sequence + 1,
            )
        except Exception as exc:
            emit_a3_trace(
                logger,
                "activity_event.failed",
                {
                    "stage_name": sanitize_error_message(
                        event.get("stage", ""),
                        max_chars=160,
                    ),
                    "error_type": type(exc).__name__,
                },
                state={"request_id": request_id, "thread_id": thread_id},
                env_flag="LOG_A3_TRACE",
            )
            return None
        return await _record_activity(activity) if activity is not None else None

    async def _finalize_stream_activity(
        *,
        status: ActivityStatus,
        title: str,
        summary: str,
        node: str = "",
        error_type: str = "",
    ) -> AgentStreamEventDraftV2 | None:
        if not activity_enabled:
            return None
        activity = build_activity_event(
            thread_id=thread_id,
            request_id=request_id,
            sequence=activity_sequence + 1,
            kind="stream",
            status=status,
            activity_key=f"stream:{request_id}",
            title=title,
            summary=summary,
            node=node,
            safe_details={"error_type": error_type} if error_type else {},
        )
        return await _record_activity(activity)

    async def _interrupt_activity_events(
        interrupt_type: str,
        *,
        node: str = "",
    ) -> list[AgentStreamEventDraftV2]:
        if not activity_enabled:
            return []
        stream_event = await _finalize_stream_activity(
            status="interrupted",
            title="Request processing interrupted",
            summary="Streaming graph execution is waiting for continuation",
            node=node,
        )
        waiting_event = await _record_activity(
            build_activity_event(
                thread_id=thread_id,
                request_id=request_id,
                sequence=activity_sequence + 1,
                kind="interrupt",
                status="waiting",
                activity_key=f"interrupt:{interrupt_type}",
                title="Waiting for user input",
                summary="Graph execution paused at a persisted interrupt",
                node=node,
                safe_details={"interrupt_type": interrupt_type},
            )
        )
        return [event for event in (stream_event, waiting_event) if event is not None]

    if activity_enabled:
        stream_started = build_activity_event(
            thread_id=thread_id,
            request_id=request_id,
            sequence=activity_sequence + 1,
            kind="stream",
            status="running",
            activity_key=f"stream:{request_id}",
            title="Request processing",
            summary="Streaming graph execution started",
        )
        yield await _record_activity(stream_started)

    def _drain_evidence_progress_events() -> list[AgentStreamEventDraftV2]:
        return [
            _stream_draft(
                "evidence_progress",
                evidence_progress_events.popleft().model_dump(mode="json"),
            )
            for _ in range(len(evidence_progress_events))
        ]

    async def _drain_trace_events() -> list[AgentStreamEventDraftV2]:
        nonlocal llm_input_manifests, manifest_state_context
        nonlocal worker_interrupted_seen, recovered_profile_completion_request
        drained: list[AgentStreamEventDraftV2] = []
        while trace_events:
            event = trace_events.pop(0)
            stage = event.get("stage")
            if is_evidence_trace_stage(stage):
                continue
            activity_payload = await _activity_from_trace(event)
            if activity_payload:
                drained.append(activity_payload)
            recovered_from_trace = _profile_completion_request_from_trace_event(event)
            if recovered_from_trace:
                recovered_profile_completion_request = recovered_from_trace
            if stage == "resource_generation.worker.interrupted":
                worker_interrupted_seen = True
            elif (
                stage == "resource_subnode.end"
                and str(event.get("status") or "") == "interrupted"
                and str(event.get("error_type") or "") == "GraphInterrupt"
            ):
                worker_interrupted_seen = True
            if isinstance(stage, str) and stage.startswith("context_"):
                request_context_events.append(_safe_context_event_summary(event))
            if stage == "provider_dispatch.started":
                if event.get("trigger_eligible") is True:
                    await _update_last_provider_dispatch_from_trace(
                        graph,
                        config,
                        thread_id=thread_id,
                        event=event,
                        state_context=manifest_state_context,
                    )
                continue
            if stage == "context_injection.dispatched":
                window_v3 = await _update_session_context_memory_from_trace(
                    graph,
                    config,
                    thread_id=thread_id,
                    event=event,
                    state_context=manifest_state_context,
                )
                drained.append(
                    _activity_update_draft(
                        "thread_context_window_v3",
                        {"thread_context_window_v3": window_v3},
                    )
                )
                continue
            if stage in WORKSPACE_TRACE_STAGES:
                payload = {
                    "type": "workspace_context",
                    **_safe_workspace_event_summary(event),
                }
                request_context_events.append(_safe_workspace_event_summary(event))
                await _update_context_window_state_from_trace(
                    graph,
                    config,
                    thread_id=thread_id,
                    request_context_events=request_context_events,
                    context_usage_history=context_usage_history,
                    last_context_policy_by_node=last_context_policy_by_node,
                    last_provider_supply_by_node=last_provider_supply_by_node,
                    last_context_selection_by_node=last_context_selection_by_node,
                    last_context_applied_by_node=last_context_applied_by_node,
                    last_drop_reasons_by_node=last_drop_reasons_by_node,
                    last_resource_subnodes=last_resource_subnodes,
                    current_node=sanitize_error_message(
                        event.get("node_name", ""),
                        max_chars=120,
                    ),
                )
                drained.append(_trace_progress_draft(payload))
                continue
            if stage in PROVIDER_RETRY_TRACE_STAGES:
                payload = {
                    "type": "provider_retry",
                    "stage": event.get("stage", ""),
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "provider": event.get("provider", ""),
                    "model": event.get("model", ""),
                    "retry_count": event.get("retry_count", 0),
                    "max_retries": event.get("max_retries", 0),
                    "next_attempt": event.get("next_attempt", 0),
                    "error_type": event.get("error_type", ""),
                    "error_message": event.get("error_message", ""),
                    "status_code": event.get("status_code"),
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage in {"resource_subnode.start", "resource_subnode.end"}:
                elapsed_ms = event.get("elapsed_ms", 0)
                payload = {
                    "type": "resource_subnode",
                    "stage": stage,
                    "resource_type": sanitize_error_message(
                        event.get("resource_type", ""),
                        max_chars=80,
                    ),
                    "subnode": sanitize_error_message(
                        event.get("subnode", ""),
                        max_chars=120,
                    ),
                    "elapsed_ms": elapsed_ms
                    if isinstance(elapsed_ms, int) and not isinstance(elapsed_ms, bool)
                    else 0,
                    "status": sanitize_error_message(
                        event.get("status", ""),
                        max_chars=40,
                    ),
                    "error_type": sanitize_error_message(
                        event.get("error_type", ""),
                        max_chars=120,
                    ),
                }
                last_resource_subnodes.append(payload)
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_policy_resolved":
                payload = _context_policy_resolved_payload(event)
                node = str(payload.get("node") or "")
                if node:
                    last_context_policy_by_node[node] = payload
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_provider_supply_plan":
                payload = _context_provider_supply_plan_payload(event)
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_provider_supply":
                payload = _context_provider_supply_payload(event)
                node = str(payload.get("node") or "")
                if node:
                    last_provider_supply_by_node[node] = payload
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_source_filter":
                payload = _context_source_filter_payload(event)
                node = str(payload.get("node") or "")
                if node:
                    last_drop_reasons_by_node[node] = payload.get("drop_reasons", {})
                drained.append(_trace_progress_draft(payload))
                continue
            if stage in {
                "context_window_state_updated",
                "context_window_state_update_failed",
            }:
                payload = {
                    "type": stage,
                    "request_id": sanitize_error_message(
                        event.get("request_id", ""),
                        max_chars=120,
                    ),
                    "current_node": sanitize_error_message(
                        event.get("current_node", ""),
                        max_chars=120,
                    ),
                    "request_event_count": event.get("request_event_count", 0),
                    "context_usage_history_count": event.get(
                        "context_usage_history_count", 0
                    ),
                    "context_usage_history_kind": sanitize_error_message(
                        event.get("context_usage_history_kind", "llm_call_history"),
                        max_chars=80,
                    ),
                    "policy_node_count": event.get("policy_node_count", 0),
                    "supply_node_count": event.get("supply_node_count", 0),
                    "selection_node_count": event.get("selection_node_count", 0),
                    "applied_node_count": event.get("applied_node_count", 0),
                    "resource_subnode_count": event.get("resource_subnode_count", 0),
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "llm_input_manifest.built":
                (
                    payload,
                    ledger_update,
                    background_window,
                    llm_input_manifests,
                ) = await _update_llm_manifest_state_from_trace(
                    graph,
                    config,
                    thread_id=thread_id,
                    event=event,
                    llm_input_manifests=llm_input_manifests,
                    state_context=manifest_state_context,
                )
                request_context_events.append(
                    {
                        "stage": "llm_input_manifest.built",
                        "node_name": payload.get("node_name", ""),
                        "llm_node": payload.get("llm_node", ""),
                        "request_id": payload.get("request_id", ""),
                        "manifest_id": payload.get("manifest_id", ""),
                        "section_count": len(payload.get("section_names") or []),
                    }
                )
                manifest_state_context.update(
                    {
                        "llm_input_manifest": payload,
                        "llm_input_manifests": list(llm_input_manifests),
                        "thread_context_ledger": ledger_update,
                        "background_context_window": background_window,
                    }
                )
                sse_payload = {
                    "type": "llm_input_manifest",
                    **payload,
                    "background_context_window": background_window,
                }
                drained.append(_trace_progress_draft(sse_payload))
                continue
            if stage == "llm_input_manifest.failed":
                payload = {
                    "type": "llm_input_manifest_error",
                    "node": sanitize_error_message(
                        event.get("node_name", ""),
                        max_chars=120,
                    ),
                    "llm_node": sanitize_error_message(
                        event.get("llm_node", ""),
                        max_chars=120,
                    ),
                    "reason": sanitize_error_message(
                        event.get("reason", ""),
                        max_chars=160,
                    ),
                    "error_type": sanitize_error_message(
                        event.get("error_type", ""),
                        max_chars=120,
                    ),
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_usage_report":
                try:
                    report = ContextUsageReport.model_validate(
                        {
                            field: event.get(field)
                            for field in ContextUsageReport.model_fields
                        }
                    )
                except Exception as exc:
                    payload = {
                        "type": "context_usage_report_error",
                        "schema_version": "context_usage_report_error_v1",
                        "manifest_id": sanitize_error_message(
                            event.get("manifest_id", ""),
                            max_chars=180,
                        ),
                        "reason": "context_usage_report_contract_invalid",
                        "warning": "context usage report contract validation failed",
                        "error_type": type(exc).__name__,
                        "node": sanitize_error_message(
                            event.get("node_name", ""),
                            max_chars=120,
                        ),
                        "node_name": sanitize_error_message(
                            event.get("node_name", ""),
                            max_chars=120,
                        ),
                        "llm_node": sanitize_error_message(
                            event.get("llm_node", ""),
                            max_chars=120,
                        ),
                        "provider": sanitize_error_message(
                            event.get("provider", ""),
                            max_chars=120,
                        ),
                        "model": sanitize_error_message(
                            event.get("model", ""),
                            max_chars=160,
                        ),
                    }
                    drained.append(_trace_progress_draft(payload))
                    continue
                report_payload = report.model_dump(mode="json")
                context_usage_reports[:] = merge_context_usage_report_history(
                    context_usage_reports,
                    [report_payload],
                )
                active_run = get_active_run(thread_id)
                active_thread_window = (
                    dict(active_run.get("thread_context_window") or {})
                    if isinstance(active_run, dict)
                    else {}
                )
                background_window = (
                    dict(active_run.get("background_context_window") or {})
                    if isinstance(active_run, dict)
                    and isinstance(active_run.get("background_context_window"), dict)
                    else dict(
                        manifest_state_context.get("background_context_window") or {}
                    )
                    if isinstance(
                        manifest_state_context.get("background_context_window"),
                        dict,
                    )
                    else {}
                )
                background_window.update(
                    {
                        "used_tokens": report.used_tokens,
                        "max_context_tokens": report.max_context_tokens,
                        "used_ratio": round(report.used_ratio, 4),
                        "updated_at": report.created_at,
                    }
                )
                active_thread_window.update(
                    {
                        "context_usage_report_count": len(context_usage_reports),
                        "context_usage_report_present": True,
                        "background_context_window": background_window,
                        **background_context_status_payload(background_window),
                    }
                )
                await _try_update_run_state(
                    graph,
                    config,
                    {
                        "context_usage_report": report_payload,
                        "context_usage_reports": [report_payload],
                        "background_context_window": background_window,
                        "thread_context_window": active_thread_window,
                    },
                    state={
                        "request_id": request_id,
                        "thread_id": thread_id,
                        "session_id": thread_id,
                    },
                    persist_checkpoint=True,
                )
                manifest_state_context.update(
                    {
                        "context_usage_report": report_payload,
                        "context_usage_reports": list(context_usage_reports),
                        "background_context_window": background_window,
                    }
                )
                payload = {
                    "type": "context_usage_report",
                    **report_payload,
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_usage_report_error":
                payload = {
                    "type": "context_usage_report_error",
                    "schema_version": "context_usage_report_error_v1",
                    "manifest_id": sanitize_error_message(
                        event.get("manifest_id", ""),
                        max_chars=180,
                    ),
                    "node": sanitize_error_message(
                        event.get("node_name", ""),
                        max_chars=120,
                    ),
                    "node_name": sanitize_error_message(
                        event.get("node_name", ""),
                        max_chars=120,
                    ),
                    "llm_node": sanitize_error_message(
                        event.get("llm_node", ""),
                        max_chars=120,
                    ),
                    "reason": sanitize_error_message(
                        event.get("reason", ""),
                        max_chars=160,
                    ),
                    "warning": sanitize_error_message(
                        event.get("warning", ""),
                        max_chars=200,
                    ),
                    "error_type": sanitize_error_message(
                        event.get("error_type", ""),
                        max_chars=120,
                    ),
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_usage":
                payload = {
                    "type": "context_usage",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "trace_call_id": event.get("trace_call_id", ""),
                    "trace_seq": event.get("trace_seq", 0),
                    "provider": event.get("provider", ""),
                    "model": event.get("model", ""),
                    "input_estimated_tokens": event.get("input_estimated_tokens", 0),
                    "reserved_output_tokens": event.get("reserved_output_tokens", 0),
                    "used_tokens": event.get("used_tokens", 0),
                    "max_context_tokens": event.get("max_context_tokens", 0),
                    "available_tokens": event.get("available_tokens", 0),
                    "used_ratio": event.get("used_ratio", 0),
                    "warning_level": event.get("warning_level", "ok"),
                    "estimated": bool(event.get("estimated", True)),
                    "tokenizer_mode": event.get("tokenizer_mode", ""),
                    "message_count": event.get("message_count", 0),
                    "schema_size_chars": event.get("schema_size_chars"),
                    "breakdown": event.get("breakdown")
                    if isinstance(event.get("breakdown"), dict)
                    else {},
                }
                context_usage_history.append(payload)
                context_usage_history[:] = trim_context_usage_history(
                    context_usage_history
                )
                active_run = get_active_run(thread_id)
                active_thread_window = (
                    dict(active_run.get("thread_context_window") or {})
                    if isinstance(active_run, dict)
                    else {}
                )
                active_thread_window["context_usage_history_count"] = len(
                    context_usage_history
                )
                background_window = (
                    dict(active_run.get("background_context_window") or {})
                    if isinstance(active_run, dict)
                    else {}
                )
                if background_window:
                    max_context_tokens = _safe_int(event.get("max_context_tokens"))
                    used_tokens = _safe_int(event.get("used_tokens"))
                    background_window.update(
                        {
                            "used_tokens": used_tokens,
                            "max_context_tokens": max_context_tokens,
                            "used_ratio": round(
                                used_tokens / max_context_tokens,
                                4,
                            )
                            if max_context_tokens > 0
                            else 0.0,
                        }
                    )
                    active_thread_window.update(
                        {
                            "background_context_window": background_window,
                            **background_context_status_payload(background_window),
                        }
                    )
                failed_update_ok = await _try_update_run_state(
                    graph,
                    config,
                    {
                        "context_usage": payload,
                        "context_usage_history": [payload],
                        "thread_context_window": active_thread_window,
                        "background_context_window": background_window
                        if background_window
                        else {},
                    },
                    state={"thread_id": thread_id, "session_id": thread_id},
                    persist_checkpoint=False,
                )
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_usage_error":
                payload = {
                    "type": "context_usage_error",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "provider": event.get("provider", ""),
                    "model": event.get("model", ""),
                    "reason": event.get("reason", ""),
                    "warning": event.get(
                        "warning", "context usage telemetry unavailable"
                    ),
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_items_collected":
                payload = {
                    "type": "context_items_collected",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "trace_call_id": event.get("trace_call_id", ""),
                    "trace_seq": event.get("trace_seq", 0),
                    "provider_count": event.get("provider_count", 0),
                    "item_count": event.get("item_count", 0),
                    "source_counts": event.get("source_counts")
                    if isinstance(event.get("source_counts"), dict)
                    else {},
                    "total_estimated_tokens": event.get("total_estimated_tokens", 0),
                    "evidence_rejected_count": event.get("evidence_rejected_count", 0),
                    "evidence_reject_reasons": _safe_int_dict(
                        event.get("evidence_reject_reasons")
                    ),
                    "missing_required_relevance_score_count": event.get(
                        "missing_required_relevance_score_count", 0
                    ),
                    "invalid_relevance_score_count": event.get(
                        "invalid_relevance_score_count", 0
                    ),
                    "top_items": _safe_context_top_items(event.get("top_items")),
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_provider_error":
                payload = {
                    "type": "context_provider_error",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "trace_call_id": event.get("trace_call_id", ""),
                    "trace_seq": event.get("trace_seq", 0),
                    "provider": event.get("provider", ""),
                    "source_type": event.get("source_type", ""),
                    "provider_stage": event.get("provider_stage", ""),
                    "error_type": event.get("error_type", ""),
                    "error_reason": event.get("error_reason", ""),
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_packing_plan":
                payload = {
                    "type": "context_packing_plan",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "trace_call_id": event.get("trace_call_id", ""),
                    "trace_seq": event.get("trace_seq", 0),
                    "candidate_count": event.get("candidate_count", 0),
                    "source_counts": event.get("source_counts")
                    if isinstance(event.get("source_counts"), dict)
                    else {},
                    "max_context_block_tokens": event.get(
                        "max_context_block_tokens", 0
                    ),
                    "strategy": event.get("strategy", ""),
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_packed":
                payload = {
                    "type": "context_packed",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "trace_call_id": event.get("trace_call_id", ""),
                    "trace_seq": event.get("trace_seq", 0),
                    "strategy": event.get("strategy", ""),
                    "selected_count": event.get("selected_count", 0),
                    "dropped_count": event.get("dropped_count", 0),
                    "selected_tokens": event.get("selected_tokens", 0),
                    "dropped_tokens": event.get("dropped_tokens", 0),
                    "required_tokens": event.get("required_tokens", 0),
                    "optional_tokens": event.get("optional_tokens", 0),
                    "remaining_tokens": event.get("remaining_tokens", 0),
                    "overflow": bool(event.get("overflow", False)),
                    "selected_items_preview": _safe_packing_preview_items(
                        event.get("selected_items_preview")
                    ),
                    "dropped_items_preview": _safe_packing_preview_items(
                        event.get("dropped_items_preview")
                    ),
                    "warnings": event.get("warnings")
                    if isinstance(event.get("warnings"), list)
                    else [],
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_packing_error":
                payload = {
                    "type": "context_packing_error",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "trace_call_id": event.get("trace_call_id", ""),
                    "trace_seq": event.get("trace_seq", 0),
                    "reason": event.get("reason", ""),
                    "warning": event.get("warning", ""),
                    "selected_tokens": event.get("selected_tokens"),
                    "budget_tokens": event.get("budget_tokens"),
                    "error_type": event.get("error_type", ""),
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_apply_plan":
                payload = {
                    "type": "context_apply_plan",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "trace_call_id": event.get("trace_call_id", ""),
                    "trace_seq": event.get("trace_seq", 0),
                    "apply_enabled": bool(event.get("apply_enabled", False)),
                    "mode": sanitize_error_message(
                        event.get("mode", ""),
                        max_chars=80,
                    ),
                    "risk_tier": event.get("risk_tier", 0),
                    "policy_source": sanitize_error_message(
                        event.get("policy_source", ""),
                        max_chars=80,
                    ),
                    "original_message_count": event.get("original_message_count", 0),
                    "selected_item_count": event.get("selected_item_count", 0),
                    "injectable_item_count": event.get("injectable_item_count", 0),
                    "skipped_item_count": event.get("skipped_item_count", 0),
                    "injection_role": event.get("injection_role", ""),
                    "injection_position": event.get("injection_position", ""),
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_apply_selection":
                payload = {
                    "type": "context_apply_selection",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "trace_call_id": event.get("trace_call_id", ""),
                    "trace_seq": event.get("trace_seq", 0),
                    "mode": sanitize_error_message(
                        event.get("mode", ""),
                        max_chars=80,
                    ),
                    "risk_tier": event.get("risk_tier", 0),
                    "policy_source": sanitize_error_message(
                        event.get("policy_source", ""),
                        max_chars=80,
                    ),
                    "skip_reason": sanitize_error_message(
                        event.get("skip_reason", ""),
                        max_chars=120,
                    ),
                    "single_resource_result": sanitize_error_message(
                        event.get("single_resource_result", ""),
                        max_chars=120,
                    ),
                    "selected_item_count": event.get("selected_item_count", 0),
                    "injectable_item_count": event.get("injectable_item_count", 0),
                    "skipped_item_count": event.get("skipped_item_count", 0),
                    "quality_filtered_count": event.get("quality_filtered_count", 0),
                    "budget_dropped_count": event.get("budget_dropped_count", 0),
                    "final_injected_count": event.get("final_injected_count", 0),
                    "injected_context_tokens": event.get("injected_context_tokens", 0),
                    "source_counts_before": _safe_int_dict(
                        event.get("source_counts_before")
                    ),
                    "source_counts_after": _safe_int_dict(
                        event.get("source_counts_after")
                    ),
                    "source_counts_dropped": _safe_int_dict(
                        event.get("source_counts_dropped")
                    ),
                    "drop_reasons": _safe_int_dict(event.get("drop_reasons")),
                    "source_drop_reasons": _safe_int_dict(
                        event.get("source_drop_reasons")
                    ),
                    "budget_drop_reasons": _safe_int_dict(
                        event.get("budget_drop_reasons")
                    ),
                    "warnings": _safe_warning_list(event.get("warnings")),
                }
                node = str(payload.get("node") or "")
                if node:
                    last_context_selection_by_node[node] = payload
                    last_drop_reasons_by_node[node] = payload.get("drop_reasons", {})
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_applied":
                payload = {
                    "type": "context_applied",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "trace_call_id": event.get("trace_call_id", ""),
                    "trace_seq": event.get("trace_seq", 0),
                    "applied": bool(event.get("applied", False)),
                    "mode": sanitize_error_message(
                        event.get("mode", ""),
                        max_chars=80,
                    ),
                    "risk_tier": event.get("risk_tier", 0),
                    "policy_source": sanitize_error_message(
                        event.get("policy_source", ""),
                        max_chars=80,
                    ),
                    "original_message_count": event.get("original_message_count", 0),
                    "final_message_count": event.get("final_message_count", 0),
                    "injected_items_count": event.get("injected_items_count", 0),
                    "skipped_items_count": event.get("skipped_items_count", 0),
                    "injected_context_tokens": event.get("injected_context_tokens", 0),
                    "budget_dropped_count": event.get("budget_dropped_count", 0),
                    "final_injected_count": event.get("final_injected_count", 0),
                    "original_estimated_tokens": event.get(
                        "original_estimated_tokens", 0
                    ),
                    "final_estimated_tokens": event.get("final_estimated_tokens", 0),
                    "token_delta": event.get("token_delta", 0),
                    "source_counts_after": _safe_int_dict(
                        event.get("source_counts_after")
                    ),
                    "drop_reasons": _safe_int_dict(event.get("drop_reasons")),
                    "source_drop_reasons": _safe_int_dict(
                        event.get("source_drop_reasons")
                    ),
                    "budget_drop_reasons": _safe_int_dict(
                        event.get("budget_drop_reasons")
                    ),
                    "injection_role": event.get("injection_role", ""),
                    "injection_position": event.get("injection_position", ""),
                    "warnings": _safe_warning_list(event.get("warnings")),
                }
                node = str(payload.get("node") or "")
                if node:
                    last_context_applied_by_node[node] = payload
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_apply_policy_resolved_summary":
                payload = {
                    "type": "context_apply_policy_resolved_summary",
                    "enabled": bool(event.get("enabled", False)),
                    "legacy_mode_enabled": bool(
                        event.get("legacy_mode_enabled", False)
                    ),
                    "legacy_global_enabled": bool(
                        event.get("legacy_global_enabled", False)
                    ),
                    "node_policy_enabled": bool(
                        event.get("node_policy_enabled", False)
                    ),
                    "node_policy_schema_configured": bool(
                        event.get("node_policy_schema_configured", False)
                    ),
                    "node_policy_count": event.get("node_policy_count", 0),
                    "node_group_count": event.get("node_group_count", 0),
                    "resource_type_policy_count": event.get(
                        "resource_type_policy_count", 0
                    ),
                    "default_policy_mode": sanitize_error_message(
                        event.get("default_policy_mode", ""),
                        max_chars=80,
                    ),
                    "default_risk_tier": event.get("default_risk_tier", 0),
                    "active_nodes": _safe_warning_list(event.get("active_nodes")),
                    "observe_only_nodes": _safe_warning_list(
                        event.get("observe_only_nodes")
                    ),
                    "disabled_nodes": _safe_warning_list(event.get("disabled_nodes")),
                    "source_defaults": _safe_warning_list(event.get("source_defaults")),
                    "importance_scoring_enabled": bool(
                        event.get("importance_scoring_enabled", False)
                    ),
                    "importance_scoring_shadow_mode": bool(
                        event.get("importance_scoring_shadow_mode", False)
                    ),
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_apply_error":
                payload = {
                    "type": "context_apply_error",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "trace_call_id": event.get("trace_call_id", ""),
                    "trace_seq": event.get("trace_seq", 0),
                    "reason": event.get("reason", ""),
                    "warning": sanitize_error_message(event.get("warning", "")),
                    "error_scope": sanitize_error_message(
                        event.get("error_scope", ""),
                        max_chars=80,
                    ),
                    "recoverable": bool(event.get("recoverable", False)),
                    "required_sources_missing": _safe_warning_list(
                        event.get("required_sources_missing")
                    ),
                    "required_sources_filtered_out": _safe_warning_list(
                        event.get("required_sources_filtered_out")
                    ),
                    "optional_sources_missing": _safe_warning_list(
                        event.get("optional_sources_missing")
                    ),
                    "provider_missing_reasons": _safe_reason_dict(
                        event.get("provider_missing_reasons")
                    ),
                    "source_drop_reasons": _safe_int_dict(
                        event.get("source_drop_reasons")
                    ),
                    "budget_drop_reasons": _safe_int_dict(
                        event.get("budget_drop_reasons")
                    ),
                    "source_counts_before": _safe_int_dict(
                        event.get("source_counts_before")
                    ),
                    "source_counts_after": _safe_int_dict(
                        event.get("source_counts_after")
                    ),
                    "source_counts_dropped": _safe_int_dict(
                        event.get("source_counts_dropped")
                    ),
                    "error_type": event.get("error_type", ""),
                }
                drained.append(_trace_progress_draft(payload))
                context_error_payload = {
                    "type": "context_error",
                    "stage": "context_apply_error",
                    "node": payload["node"],
                    "llm_node": payload["llm_node"],
                    "trace_call_id": payload["trace_call_id"],
                    "trace_seq": payload["trace_seq"],
                    "reason": payload["reason"],
                    "required_sources_missing": payload["required_sources_missing"],
                    "required_sources_filtered_out": payload[
                        "required_sources_filtered_out"
                    ],
                    "recoverable": payload["recoverable"],
                    "provider_missing_reasons": payload["provider_missing_reasons"],
                    "source_drop_reasons": payload["source_drop_reasons"],
                    "budget_drop_reasons": payload["budget_drop_reasons"],
                    "source_counts_before": payload["source_counts_before"],
                    "source_counts_after": payload["source_counts_after"],
                    "source_counts_dropped": payload["source_counts_dropped"],
                }
                drained.append(_trace_progress_draft(context_error_payload))
                failed_update_ok = await _try_update_run_state(
                    graph,
                    config,
                    {
                        "run_status": RUN_STATUS_ERROR,
                        "resume_available": False,
                        "pending_interrupt_type": "",
                    },
                    state={"thread_id": thread_id, "session_id": thread_id},
                )
                if failed_update_ok:
                    finish_active_run(thread_id, {"run_status": RUN_STATUS_ERROR})
                await _update_context_window_state_from_trace(
                    graph,
                    config,
                    thread_id=thread_id,
                    request_context_events=request_context_events,
                    context_usage_history=context_usage_history,
                    last_context_policy_by_node=last_context_policy_by_node,
                    last_provider_supply_by_node=last_provider_supply_by_node,
                    last_context_selection_by_node=last_context_selection_by_node,
                    last_context_applied_by_node=last_context_applied_by_node,
                    last_drop_reasons_by_node=last_drop_reasons_by_node,
                    last_resource_subnodes=last_resource_subnodes,
                    current_node=str(payload["node"] or ""),
                )
                window_payload = {
                    "type": "context_window_state_updated",
                    "node": payload["node"],
                    "llm_node": payload["llm_node"],
                    "request_event_count": len(request_context_events),
                }
                drained.append(_trace_progress_draft(window_payload))
                continue
            if stage == "plain_llm_output" or stage == "structured_llm_output":
                await _update_context_window_state_from_trace(
                    graph,
                    config,
                    thread_id=thread_id,
                    request_context_events=request_context_events,
                    context_usage_history=context_usage_history,
                    last_context_policy_by_node=last_context_policy_by_node,
                    last_provider_supply_by_node=last_provider_supply_by_node,
                    last_context_selection_by_node=last_context_selection_by_node,
                    last_context_applied_by_node=last_context_applied_by_node,
                    last_drop_reasons_by_node=last_drop_reasons_by_node,
                    last_resource_subnodes=last_resource_subnodes,
                    current_node=str(event.get("node_name") or ""),
                )
                payload = {
                    "type": "context_window_state_updated",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "request_event_count": len(request_context_events),
                }
                drained.append(_trace_progress_draft(payload))
                continue
            if stage == "context_importance_scored":
                payload = {
                    "type": "context_importance_scored",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "source_counts": _safe_int_dict(event.get("source_counts")),
                    "score_buckets": _safe_int_dict(event.get("score_buckets")),
                    "reason_code_counts": _safe_int_dict(
                        event.get("reason_code_counts")
                    ),
                    "candidate_count": event.get("candidate_count", 0),
                    "scored_count": event.get("scored_count", 0),
                    "kept_count": event.get("kept_count", 0),
                    "dropped_count": event.get("dropped_count", 0),
                    "scoring_elapsed_ms": event.get("scoring_elapsed_ms", 0),
                    "disabled_reason": sanitize_error_message(
                        event.get("disabled_reason", ""),
                        max_chars=160,
                    ),
                    "error_reason": sanitize_error_message(
                        event.get("error_reason", ""),
                        max_chars=160,
                    ),
                    "error_type": sanitize_error_message(
                        event.get("error_type", ""),
                        max_chars=120,
                    ),
                    "warnings": _safe_warning_list(event.get("warnings")),
                }
                drained.append(_trace_progress_draft(payload))
                continue
        return drained

    def _drain_provisional_events() -> list[AgentStreamEventDraftV2]:
        nonlocal provisional_block_open
        drained: list[AgentStreamEventDraftV2] = []
        block_id = f"{request_id}:qa-answer"
        while provisional_events:
            event = provisional_events.pop(0)
            event_type = str(event.get("type") or "")
            if event_type == "qa_provisional_start":
                if not provisional_block_open:
                    provisional_block_open = True
                    drained.append(
                        _content_block_draft(
                            "content_block_start",
                            block_id=block_id,
                            block_index=0,
                        )
                    )
                continue
            if event_type == "qa_provisional_delta":
                if not provisional_block_open:
                    provisional_block_open = True
                    drained.append(
                        _content_block_draft(
                            "content_block_start",
                            block_id=block_id,
                            block_index=0,
                        )
                    )
                drained.append(
                    _content_block_draft(
                        "content_block_delta",
                        block_id=block_id,
                        block_index=0,
                        delta=str(event.get("delta") or ""),
                    )
                )
                continue
            if event_type in {"qa_provisional_stop", "qa_provisional_reset"}:
                if provisional_block_open:
                    provisional_block_open = False
                    drained.append(
                        _content_block_draft(
                            "content_block_stop",
                            block_id=block_id,
                            block_index=0,
                            reset=event_type == "qa_provisional_reset",
                            reason=str(event.get("reason") or ""),
                        )
                    )
                continue
            raise StreamContractError("unknown provisional stream event")
        return drained

    def _close_open_content_blocks(
        *,
        reset: bool,
        reason: str,
    ) -> list[AgentStreamEventDraftV2]:
        nonlocal provisional_block_open
        drained: list[AgentStreamEventDraftV2] = []
        if provisional_block_open:
            provisional_block_open = False
            drained.append(
                _content_block_draft(
                    "content_block_stop",
                    block_id=f"{request_id}:qa-answer",
                    block_index=0,
                    reset=reset,
                    reason=reason,
                )
            )
        for node_name, (block_id, block_index) in list(node_content_blocks.items()):
            node_content_blocks.pop(node_name, None)
            drained.append(
                _content_block_draft(
                    "content_block_stop",
                    block_id=block_id,
                    block_index=block_index,
                    reset=reset,
                    reason=reason,
                )
            )
        return drained

    try:
        async for event in graph.astream_events(
            input_data, config=config, version="v2"
        ):
            for provisional_payload in _drain_provisional_events():
                yield provisional_payload
            for progress_payload in _drain_evidence_progress_events():
                yield progress_payload
            for trace_payload in await _drain_trace_events():
                yield trace_payload
            event_type = event["event"]

            # Node lifecycle events
            if event_type in ("on_chain_start", "on_chain_end"):
                node_name = event.get("name")
                meta_node = event.get("metadata", {}).get("langgraph_node")
                # Only emit for top-level graph nodes (name matches metadata),
                # not for internal sub-chains (RunnableSequence, etc.).
                if (
                    node_name
                    and node_name == meta_node
                    and node_name in runtime_graph_nodes
                ):
                    activity_draft = None
                    if event_type == "on_chain_start":
                        node_start_times[node_name] = time.monotonic()
                        if node_name not in active_nodes:
                            active_nodes.append(node_name)
                        await _try_update_run_state(
                            graph,
                            config,
                            {
                                "schema_version": RUN_CONTROL_SCHEMA_VERSION,
                                "run_status": RUN_STATUS_RUNNING,
                                "current_node": node_name,
                                "pending_interrupt_type": "",
                            },
                            state={"thread_id": thread_id, "session_id": thread_id},
                            persist_checkpoint=False,
                        )
                        payload = {
                            "status": "start",
                            "node": node_name,
                        }
                        if activity_enabled:
                            activity_draft = await _record_activity(
                                build_node_activity_event(
                                    thread_id=thread_id,
                                    request_id=request_id,
                                    sequence=activity_sequence + 1,
                                    node_id=node_name,
                                    status="running",
                                )
                            )
                    else:
                        duration_ms = None
                        start_t = node_start_times.pop(node_name, None)
                        if node_name in active_nodes:
                            active_nodes.remove(node_name)
                        if start_t is not None:
                            duration_ms = round((time.monotonic() - start_t) * 1000)

                        error = None
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict) and output.get("error"):
                            error = str(output["error"])
                        if (
                            node_name == "resource_bundle_output"
                            and isinstance(output, dict)
                            and isinstance(output.get("resource_final_v3"), dict)
                            and bool(output.get("resource_final_v3"))
                        ):
                            terminal_resource_output = output

                        payload = {
                            "status": "end",
                            "node": node_name,
                            "duration_ms": duration_ms,
                            "error": error,
                        }
                        if activity_enabled:
                            activity_draft = await _record_activity(
                                build_node_activity_event(
                                    thread_id=thread_id,
                                    request_id=request_id,
                                    sequence=activity_sequence + 1,
                                    node_id=node_name,
                                    status="failed" if error else "completed",
                                    duration_ms=duration_ms,
                                    error_type="NodeOutputError" if error else "",
                                ),
                                persist_checkpoint=True,
                            )
                        await _try_update_run_state(
                            graph,
                            config,
                            {
                                "last_completed_node": node_name,
                                "current_node": "",
                            },
                            state={"thread_id": thread_id, "session_id": thread_id},
                            persist_checkpoint=False,
                        )
                    if activity_draft:
                        yield activity_draft
                    yield _activity_update_draft("node_event", payload)

                    if (
                        event_type == "on_chain_end"
                        and node_name in node_content_blocks
                    ):
                        block_id, block_index = node_content_blocks.pop(node_name)
                        yield _content_block_draft(
                            "content_block_stop",
                            block_id=block_id,
                            block_index=block_index,
                        )

                    # Emit provisional content blocks for non-streaming answer nodes.
                    if (
                        event_type == "on_chain_end"
                        and _node_stream_mode(node_name) == "final_message"
                    ):
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict):
                            for msg in output.get("messages", []):
                                if hasattr(msg, "content") and msg.content:
                                    block_index = next_content_block_index
                                    next_content_block_index += 1
                                    block_id = (
                                        f"{request_id}:assistant:{node_name}:"
                                        f"{block_index}"
                                    )
                                    yield _content_block_draft(
                                        "content_block_start",
                                        block_id=block_id,
                                        block_index=block_index,
                                    )
                                    yield _content_block_draft(
                                        "content_block_delta",
                                        block_id=block_id,
                                        block_index=block_index,
                                        delta=str(msg.content),
                                    )
                                    yield _content_block_draft(
                                        "content_block_stop",
                                        block_id=block_id,
                                        block_index=block_index,
                                    )

                    for progress_payload in _drain_evidence_progress_events():
                        yield progress_payload
                    for trace_payload in await _drain_trace_events():
                        yield trace_payload

            # Provider token streaming becomes provisional content blocks.
            elif event_type == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node")
                if (
                    node_name in runtime_graph_nodes
                    and _node_stream_mode(node_name) == "provider_delta"
                ):
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        if node_name not in node_content_blocks:
                            block_index = next_content_block_index
                            next_content_block_index += 1
                            block_id = (
                                f"{request_id}:assistant:{node_name}:{block_index}"
                            )
                            node_content_blocks[node_name] = (block_id, block_index)
                            yield _content_block_draft(
                                "content_block_start",
                                block_id=block_id,
                                block_index=block_index,
                            )
                        block_id, block_index = node_content_blocks[node_name]
                        yield _content_block_draft(
                            "content_block_delta",
                            block_id=block_id,
                            block_index=block_index,
                            delta=str(chunk.content),
                        )

            # Token usage events
            elif event_type == "on_chat_model_end":
                node_name = event.get("metadata", {}).get("langgraph_node")
                output = event.get("data", {}).get("output")
                usage = getattr(output, "usage_metadata", None)
                if usage and node_name:
                    yield _activity_update_draft(
                        "usage",
                        {
                            "node": node_name,
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                    )
        for provisional_payload in _drain_provisional_events():
            yield provisional_payload
        for progress_payload in _drain_evidence_progress_events():
            yield progress_payload
        for trace_payload in await _drain_trace_events():
            yield trace_payload
        for block_payload in _close_open_content_blocks(
            reset=False,
            reason="graph_completed",
        ):
            yield block_payload
    except Exception as e:
        for provisional_payload in _drain_provisional_events():
            yield provisional_payload
        for progress_payload in _drain_evidence_progress_events():
            yield progress_payload
        for trace_payload in await _drain_trace_events():
            yield trace_payload
        for block_payload in _close_open_content_blocks(
            reset=True,
            reason="graph_failed",
        ):
            yield block_payload
        safe_error_message = sanitize_error_message(e, max_chars=300)
        logger.error(
            "Unhandled error in graph streaming: error_type=%s error_message=%s",
            type(e).__name__,
            safe_error_message,
        )
        failed_activity_draft = None
        if activity_enabled:
            failed_activity_draft = await _record_activity(
                build_activity_event(
                    thread_id=thread_id,
                    request_id=request_id,
                    sequence=activity_sequence + 1,
                    kind="stream",
                    status="failed",
                    activity_key=f"stream:{request_id}",
                    title="Request processing failed",
                    summary="Streaming graph execution failed",
                    node=active_nodes[-1] if active_nodes else "",
                    safe_details={"error_type": type(e).__name__},
                )
            )
        failed_update_ok = await _try_update_run_state(
            graph,
            config,
            {
                "run_status": RUN_STATUS_ERROR,
                "resume_available": False,
                "pending_interrupt_type": "",
                "activity_timeline": activity_timeline,
            },
            state={"thread_id": thread_id, "session_id": thread_id},
        )
        run_control_registry.clear_stop_signal(thread_id)
        failed_node = active_nodes[-1] if active_nodes else None
        if failed_activity_draft:
            yield failed_activity_draft
        if failed_node:
            start_t = node_start_times.get(failed_node)
            duration_ms = (
                round((time.monotonic() - start_t) * 1000)
                if start_t is not None
                else None
            )
            yield _activity_update_draft(
                "node_event",
                {
                    "status": "end",
                    "node": failed_node,
                    "duration_ms": duration_ms,
                    "error_type": type(e).__name__,
                    "synthetic": True,
                },
            )
        yield _stream_draft(
            "stream_error",
            {
                "error_type": type(e).__name__,
                "message": "Graph execution failed",
                "failed_node": failed_node,
                "active_nodes": active_nodes,
                "recoverable": False,
            },
        )
        if failed_update_ok:
            finish_active_run(thread_id, {"run_status": RUN_STATUS_ERROR})
        return
    finally:
        reset_evidence_progress_sink(evidence_progress_sink_token)
        reset_provisional_event_sink(provisional_sink_token)
        reset_trace_event_sink(trace_sink_token)

    # Check for interrupt after stream completes
    try:
        state_snapshot = await graph.aget_state(config)
    except Exception as exc:
        logger.error(
            "Failed to read graph state snapshot after stream: "
            "error_type=%s error_message=%s",
            type(exc).__name__,
            sanitize_error_message(exc, max_chars=300),
        )
        raise

    interrupt_values = _pending_interrupt_values(state_snapshot)
    snapshot_tasks = list(getattr(state_snapshot, "tasks", ()) or ())
    snapshot_next = getattr(state_snapshot, "next", ()) or ()
    task_interrupt_counts = [
        len(list(getattr(task, "interrupts", ()) or ())) for task in snapshot_tasks
    ]
    emit_a3_trace(
        logger,
        "sse_pending_interrupt_probe",
        {
            "pending_interrupt_count": len(interrupt_values),
            "state_snapshot_next": [str(item) for item in snapshot_next],
            "task_count": len(snapshot_tasks),
            "task_interrupt_counts": task_interrupt_counts,
            "pending_interrupt_types": [
                str(value.get("type") or "")
                for value in interrupt_values
                if isinstance(value, dict)
            ],
        },
        state={
            "request_id": request_id,
            "thread_id": thread_id,
            "session_id": thread_id,
        },
        env_flag="LOG_A3_TRACE",
    )

    final_state = _state_values(state_snapshot)
    # TEMP A3_TRACE: remove after state snapshot validation.
    emit_a3_trace(
        logger,
        "sse_state_snapshot",
        {
            "success": True,
            "final_state_keys": sorted(final_state.keys()),
            "has_mindmap_artifact": bool(final_state.get("mindmap_artifact")),
            "has_mindmap_tree": bool(final_state.get("mindmap_tree")),
            "has_exercise_items": bool(final_state.get("exercise_items")),
            "has_review_doc_artifact": bool(final_state.get("review_doc_artifact")),
            "has_review_doc_artifacts": bool(final_state.get("review_doc_artifacts")),
            "review_doc_artifacts_count": len(
                final_state.get("review_doc_artifacts") or []
            ),
            "exercise_items_count": len(final_state.get("exercise_items") or []),
            "requested_resource_type": final_state.get("requested_resource_type", ""),
        },
        state={
            **final_state,
            "request_id": request_id,
            "thread_id": thread_id,
            "session_id": thread_id,
        },
        env_flag="LOG_A3_TRACE",
    )

    if interrupt_values:
        interrupt_value = interrupt_values[0]
        if (
            isinstance(interrupt_value, dict)
            and interrupt_value.get("type") == "user_stop"
        ):
            stopped_at = utc_now_iso()
            stopped_node = str(interrupt_value.get("node") or "")
            interrupt_activity_payloads = await _interrupt_activity_events(
                "user_stop",
                node=stopped_node,
            )
            stopped_update_ok = await _try_update_run_state(
                graph,
                config,
                {
                    "schema_version": RUN_CONTROL_SCHEMA_VERSION,
                    "run_status": RUN_STATUS_STOPPED,
                    "resume_available": True,
                    "pending_interrupt_type": "user_stop",
                    "stopped_at": stopped_at,
                    "current_node": stopped_node,
                    "stop_reason": str(interrupt_value.get("reason") or "user_stop"),
                    "activity_timeline": activity_timeline,
                },
                state=final_state,
            )
            emit_a3_trace(
                logger,
                "run_stopped_at_checkpoint",
                {
                    "thread_id": thread_id,
                    "node": stopped_node,
                    "stopped_at": stopped_at,
                    "resume_available": True,
                },
                state=final_state,
                env_flag="LOG_A3_TRACE",
            )
            for activity_payload in interrupt_activity_payloads:
                yield activity_payload
            yield _stream_draft(
                "stopped",
                {
                    "run_status": RUN_STATUS_STOPPED,
                    "thread_id": thread_id,
                    "resume_available": True,
                    "pending_interrupt_type": "user_stop",
                    "node": stopped_node,
                    "stopped_at": stopped_at,
                },
            )
            if stopped_update_ok:
                finish_active_run(thread_id, {"run_status": RUN_STATUS_STOPPED})
            return
        if (
            isinstance(interrupt_value, dict)
            and interrupt_value.get("type") == "memory_confirmation"
        ):
            interrupt_activity_payloads = await _interrupt_activity_events(
                "memory_confirmation"
            )
            interrupt_update_ok = await _try_update_run_state(
                graph,
                config,
                {
                    "resume_available": False,
                    "pending_interrupt_type": "memory_confirmation",
                    "activity_timeline": activity_timeline,
                },
                state=final_state,
                persist_checkpoint=True,
            )
            payload_data = {
                "type": "interrupt",
                "interrupt_type": "memory_confirmation",
                "question": interrupt_value.get("question", ""),
                "reason": interrupt_value.get("reason", ""),
                "selected_memory_count": interrupt_value.get(
                    "selected_memory_count", 0
                ),
                "options": interrupt_value.get("options", []),
                "thread_id": thread_id,
            }
        elif (
            isinstance(interrupt_value, dict)
            and interrupt_value.get("type") == "profile_completion_required"
        ):
            profile_request = _safe_profile_completion_request(interrupt_value)
            interrupt_activity_payloads = await _interrupt_activity_events(
                "profile_completion_required"
            )
            interrupt_update_ok = await _try_update_run_state(
                graph,
                config,
                {
                    "resume_available": True,
                    "pending_interrupt_type": "profile_completion_required",
                    "profile_completion_request": profile_request,
                    "activity_timeline": activity_timeline,
                },
                state=final_state,
                persist_checkpoint=True,
            )
            payload_data = {
                "type": "interrupt",
                "interrupt_type": "profile_completion_required",
                "title": profile_request.get("title", ""),
                "fields": profile_request.get("fields", []),
                "profile_completion_request": profile_request,
                "resume_available": True,
                "thread_id": thread_id,
            }
        else:
            interrupt_activity_payloads = await _interrupt_activity_events(
                "plan_review"
            )
            interrupt_update_ok = await _try_update_run_state(
                graph,
                config,
                {
                    "resume_available": False,
                    "pending_interrupt_type": "plan_review",
                    "activity_timeline": activity_timeline,
                },
                state=final_state,
                persist_checkpoint=True,
            )
            payload_data = {
                "type": "interrupt",
                "interrupt_type": "plan_review",
                "draft": interrupt_value.get("value", interrupt_value),
                "thread_id": thread_id,
            }
        for activity_payload in interrupt_activity_payloads:
            yield activity_payload
        yield _stream_draft(
            "interrupt",
            {key: value for key, value in payload_data.items() if key != "type"},
        )
        if interrupt_update_ok:
            finish_active_run(thread_id, {"run_status": RUN_STATUS_STOPPED})
        return

    if worker_interrupted_seen:
        recovered_request = _complete_profile_completion_request(
            recovered_profile_completion_request
        ) or _complete_profile_completion_request(
            final_state.get("profile_completion_request")
        )
        if recovered_request:
            interrupt_activity_payloads = await _interrupt_activity_events(
                "profile_completion_required"
            )
            interrupt_update_ok = await _try_update_run_state(
                graph,
                config,
                {
                    "resume_available": True,
                    "pending_interrupt_type": "profile_completion_required",
                    "profile_completion_request": recovered_request,
                    "activity_timeline": activity_timeline,
                },
                state=final_state,
                persist_checkpoint=True,
            )
            payload_data = {
                "type": "interrupt",
                "interrupt_type": "profile_completion_required",
                "title": recovered_request.get("title", ""),
                "fields": recovered_request.get("fields", []),
                "profile_completion_request": recovered_request,
                "resume_available": True,
                "thread_id": thread_id,
            }
            emit_a3_trace(
                logger,
                "sse_pending_interrupt_recovered",
                {
                    "thread_id": thread_id,
                    "interrupt_type": "profile_completion_required",
                    "worker_interrupted_seen": True,
                    "pending_interrupt_count": 0,
                    "profile_field_count": len(recovered_request.get("fields") or []),
                },
                state=final_state,
                env_flag="LOG_A3_TRACE",
            )
            for activity_payload in interrupt_activity_payloads:
                yield activity_payload
            yield _stream_draft(
                "interrupt",
                {key: value for key, value in payload_data.items() if key != "type"},
            )
            if interrupt_update_ok:
                finish_active_run(thread_id, {"run_status": RUN_STATUS_STOPPED})
            return

        interrupt_lost_payload = {
            "type": "error",
            "error_type": "interrupt_lost",
            "message": (
                "Resource worker was interrupted, but no checkpoint interrupt was "
                "available. Completion was blocked."
            ),
            "terminal_non_completed": True,
            "thread_id": thread_id,
        }
        emit_a3_trace(
            logger,
            "sse_interrupt_lost",
            {
                "thread_id": thread_id,
                "worker_interrupted_seen": True,
                "pending_interrupt_count": 0,
                "has_recoverable_profile_completion_request": False,
            },
            state=final_state,
            env_flag="LOG_A3_TRACE",
        )
        lost_activity_payload = await _finalize_stream_activity(
            status="failed",
            title="Request processing failed",
            summary="An expected interrupt checkpoint could not be recovered",
            error_type="InterruptLostError",
        )
        await _try_update_run_state(
            graph,
            config,
            {
                "run_status": RUN_STATUS_ERROR,
                "resume_available": False,
                "pending_interrupt_type": "",
                "activity_timeline": activity_timeline,
            },
            state=final_state,
            persist_checkpoint=True,
        )
        if lost_activity_payload:
            yield lost_activity_payload
        yield _stream_draft(
            "stream_error",
            {
                key: value
                for key, value in interrupt_lost_payload.items()
                if key != "type"
            },
        )
        finish_active_run(
            thread_id,
            {
                "run_status": RUN_STATUS_ERROR,
                "error_type": "interrupt_lost",
            },
        )
        return

    if snapshot_next:
        pending_nodes = [
            sanitize_error_message(item, max_chars=120) for item in snapshot_next
        ]
        pending_checkpoint_payload = {
            "type": "error",
            "error_type": "pending_checkpoint_without_interrupt",
            "message": (
                "Graph execution ended with pending checkpoint nodes but no "
                "persisted interrupt. Completion was blocked."
            ),
            "terminal_non_completed": True,
            "thread_id": thread_id,
            "pending_nodes": pending_nodes,
        }
        emit_a3_trace(
            logger,
            "sse_pending_checkpoint_without_interrupt",
            {
                "thread_id": thread_id,
                "pending_node_count": len(pending_nodes),
                "pending_nodes": pending_nodes,
            },
            state=final_state,
            env_flag="LOG_A3_TRACE",
        )
        pending_activity_payload = await _finalize_stream_activity(
            status="failed",
            title="Request processing blocked",
            summary="Graph execution still has pending checkpoint nodes",
            error_type="PendingCheckpointWithoutInterruptError",
        )
        await _try_update_run_state(
            graph,
            config,
            {
                "run_status": RUN_STATUS_ERROR,
                "resume_available": False,
                "pending_interrupt_type": "",
                "activity_timeline": activity_timeline,
            },
            state=final_state,
            persist_checkpoint=True,
        )
        if pending_activity_payload:
            yield pending_activity_payload
        yield _stream_draft(
            "stream_error",
            {
                key: value
                for key, value in pending_checkpoint_payload.items()
                if key != "type"
            },
        )
        finish_active_run(
            thread_id,
            {
                "run_status": RUN_STATUS_ERROR,
                "error_type": "pending_checkpoint_without_interrupt",
            },
        )
        return

    final_request_context_window = {
        "current_request_id": request_context_events[-1].get("request_id", "")
        if request_context_events
        else str(final_state.get("request_id") or ""),
        "current_node": "",
        "last_event_count": len(request_context_events),
    }
    runtime_final_state = {
        **final_state,
        "thread_id": thread_id,
        "request_id": request_id,
    }
    try:
        qa_payload = qa_final_payload(final_state)
        resource_payload = _resource_final_payload(runtime_final_state)
        recommendation_payload = _recommendation_final_payload(runtime_final_state)
        terminal_resource_state = runtime_final_state
        if terminal_resource_output:
            terminal_resource_state = {
                **terminal_resource_output,
                "thread_id": thread_id,
                "request_id": request_id,
            }
            terminal_resource_payload = _resource_final_payload(terminal_resource_state)
            if terminal_resource_payload:
                resource_payload = terminal_resource_payload
    except Exception as exc:
        failed_activity_payload = await _finalize_stream_activity(
            status="failed",
            title="Authoritative terminal validation failed",
            summary="Graph terminal payload did not pass strict validation",
            error_type=type(exc).__name__,
        )
        error_update_ok = await _try_update_run_state(
            graph,
            config,
            {
                "run_status": RUN_STATUS_ERROR,
                "resume_available": False,
                "pending_interrupt_type": "",
                "current_node": "",
                "stop_requested": False,
                "request_context_window": final_request_context_window,
                "activity_timeline": activity_timeline,
            },
            state=final_state,
            persist_checkpoint=True,
        )
        run_control_registry.clear_stop_signal(thread_id)
        emit_a3_trace(
            logger,
            "sse_authoritative_terminal_invalid",
            {"error_type": type(exc).__name__},
            state=runtime_final_state,
            env_flag="LOG_A3_TRACE",
        )
        if failed_activity_payload:
            yield failed_activity_payload
        yield _stream_draft(
            "stream_error",
            {
                "error_type": "authoritative_terminal_invalid",
                "message": "Graph terminal payload failed strict validation",
                "recoverable": False,
                "terminal_non_completed": True,
            },
        )
        if error_update_ok:
            finish_active_run(
                thread_id,
                {
                    "run_status": RUN_STATUS_ERROR,
                    "error_type": "authoritative_terminal_invalid",
                },
            )
        return
    diagnostic_state = terminal_resource_state
    terminal_payload_count = sum(
        payload is not None
        for payload in (qa_payload, resource_payload, recommendation_payload)
    )
    if terminal_payload_count > 1:
        failed_activity_payload = await _finalize_stream_activity(
            status="failed",
            title="Authoritative terminal conflict",
            summary="Graph execution produced multiple authoritative finals",
        )
        error_update_ok = await _try_update_run_state(
            graph,
            config,
            {
                "run_status": RUN_STATUS_ERROR,
                "resume_available": False,
                "pending_interrupt_type": "",
                "current_node": "",
                "stop_requested": False,
                "request_context_window": final_request_context_window,
                "activity_timeline": activity_timeline,
            },
            state=final_state,
            persist_checkpoint=True,
        )
        run_control_registry.clear_stop_signal(thread_id)
        if failed_activity_payload:
            yield failed_activity_payload
        yield _stream_draft(
            "stream_error",
            {
                "error_type": "authoritative_terminal_conflict",
                "message": "Graph execution produced multiple authoritative finals",
                "recoverable": False,
                "terminal_non_completed": True,
            },
        )
        if error_update_ok:
            finish_active_run(
                thread_id,
                {
                    "run_status": RUN_STATUS_ERROR,
                    "error_type": "authoritative_terminal_conflict",
                },
            )
        return
    if not resource_payload and resource_final_v3_required(diagnostic_state):
        failed_activity_payload = await _finalize_stream_activity(
            status="failed",
            title="Resource finalization failed",
            summary="Resource workflow ended without Resource Final V3",
        )
        error_values = {
            "run_status": RUN_STATUS_ERROR,
            "resume_available": False,
            "pending_interrupt_type": "",
            "profile_completion_request": {},
            "current_node": "",
            "stop_requested": False,
            "request_context_window": final_request_context_window,
            "activity_timeline": activity_timeline,
        }
        error_update_ok = await _try_update_run_state(
            graph,
            config,
            error_values,
            state=final_state,
            persist_checkpoint=True,
        )
        run_control_registry.clear_stop_signal(thread_id)
        if failed_activity_payload:
            yield failed_activity_payload
        requested_resource_types = list(
            requested_resource_kinds_from_state(diagnostic_state)
        )
        emit_a3_trace(
            logger,
            "sse_resource_final_missing",
            {
                "error_type": "resource_final_v3_missing",
                "requested_resource_types": requested_resource_types,
                "resource_generation_status": str(
                    diagnostic_state.get("resource_generation_status") or ""
                ),
            },
            state=runtime_final_state,
            env_flag="LOG_A3_TRACE",
        )
        yield _stream_draft(
            "stream_error",
            {
                "error_type": "resource_final_v3_missing",
                "message": "Resource workflow ended without Resource Final V3",
                "recoverable": False,
                "terminal_non_completed": True,
                "requested_resource_types": requested_resource_types,
            },
        )
        if error_update_ok:
            finish_active_run(
                thread_id,
                {
                    "run_status": RUN_STATUS_ERROR,
                    "error_type": "resource_final_v3_missing",
                },
            )
        return
    final_activity_payloads: list[AgentStreamEventDraftV2] = []
    if activity_enabled and resource_payload:
        resource_terminal_status = str(resource_payload["terminal_status"])
        resource_final_identity = str(resource_payload["resource_final_id"])
        v3_resources = [
            item
            for item in (resource_payload.get("resources") or [])
            if isinstance(item, dict)
        ]
        artifact_activity = build_activity_event(
            thread_id=thread_id,
            request_id=request_id,
            sequence=activity_sequence + 1,
            kind="artifact",
            status="failed" if resource_terminal_status == "failed" else "completed",
            activity_key=f"resource_final:{resource_final_identity}",
            title="Generated resource ready",
            summary="Renderable resource payload finalized",
            node="resource_bundle_output",
            safe_details={
                "resource_final_id": resource_final_identity,
                "resource_count": len(v3_resources),
                "terminal_status": resource_terminal_status,
            },
        )
        final_activity_payloads.append(await _record_activity(artifact_activity))
    completed_activity_payload = await _finalize_stream_activity(
        status="completed",
        title="Request processing completed",
        summary="Streaming graph execution completed",
    )
    if completed_activity_payload:
        final_activity_payloads.append(completed_activity_payload)
    completed_values = {
        "run_status": RUN_STATUS_COMPLETED,
        "resume_available": False,
        "pending_interrupt_type": "",
        "profile_completion_request": {},
        "current_node": "",
        "stop_requested": False,
        "request_context_window": final_request_context_window,
        "context_window_events": list(request_context_events),
        "last_context_policy_by_node": dict(last_context_policy_by_node),
        "last_provider_supply_by_node": dict(last_provider_supply_by_node),
        "last_context_selection_by_node": dict(last_context_selection_by_node),
        "last_context_applied_by_node": dict(last_context_applied_by_node),
        "last_drop_reasons_by_node": dict(last_drop_reasons_by_node),
        "last_resource_subnodes": list(last_resource_subnodes),
        "activity_timeline": activity_timeline,
    }
    if resource_payload:
        completed_values["last_resource_final_payload"] = resource_payload
    if recommendation_payload:
        completed_values["last_recommendation_final_payload"] = recommendation_payload
    if qa_payload:
        completed_values["last_qa_response"] = qa_payload
    if context_usage_history:
        completed_values["context_usage"] = context_usage_history[-1]
        completed_values["context_usage_history"] = list(context_usage_history)
    if context_usage_reports:
        completed_values["context_usage_report"] = context_usage_reports[0]
        completed_values["context_usage_reports"] = list(context_usage_reports)
    if llm_input_manifests:
        completed_values["llm_input_manifest"] = llm_input_manifests[0]
        completed_values["llm_input_manifests"] = list(llm_input_manifests)
        background_window = manifest_state_context.get("background_context_window")
        if isinstance(background_window, dict):
            completed_values["background_context_window"] = background_window
            raw_thread_window = completed_values.get("thread_context_window")
            active_thread_window = (
                dict(raw_thread_window) if isinstance(raw_thread_window, dict) else {}
            )
            active_thread_window.update(
                {
                    "llm_input_manifest_count": len(llm_input_manifests),
                    "background_context_window": background_window,
                    **background_context_status_payload(background_window),
                }
            )
            completed_values["thread_context_window"] = active_thread_window
        ledger = manifest_state_context.get("thread_context_ledger")
        if isinstance(ledger, dict):
            completed_values["thread_context_ledger"] = ledger

    session_memory_ledger = manifest_state_context.get("session_context_memory_ledger")
    if isinstance(session_memory_ledger, dict) and session_memory_ledger:
        completed_values["session_context_memory_ledger"] = session_memory_ledger
        completed_values["thread_context_window_v3"] = _thread_context_window_v3(
            manifest_state_context,
            thread_id=thread_id,
            updating=False,
        )

    active_run = get_active_run(thread_id)
    completed_thread_window = (
        dict(active_run.get("thread_context_window") or {})
        if isinstance(active_run, dict)
        else {}
    )
    completed_thread_window.update(
        {
            "context_usage_report_count": len(context_usage_reports),
            "context_usage_report_present": bool(context_usage_reports),
            **activity_timeline_status(activity_timeline),
        }
    )
    completed_values["thread_context_window"] = completed_thread_window

    completed_update_ok = await _try_update_run_state(
        graph,
        config,
        completed_values,
        state=final_state,
        update_active_on_failure=False,
    )
    if not completed_update_ok:
        failed_activity_payload = await _finalize_stream_activity(
            status="failed",
            title="Terminal checkpoint persistence failed",
            summary="Authoritative final was not committed to the thread checkpoint",
            error_type="terminal_checkpoint_persist_failed",
        )
        await _try_update_run_state(
            graph,
            config,
            {
                "run_status": RUN_STATUS_ERROR,
                "resume_available": False,
                "pending_interrupt_type": "",
                "current_node": "",
                "stop_requested": False,
                "request_context_window": final_request_context_window,
                "activity_timeline": activity_timeline,
            },
            state=final_state,
            persist_checkpoint=True,
        )
        run_control_registry.clear_stop_signal(thread_id)
        emit_a3_trace(
            logger,
            "sse_terminal_checkpoint_persist_failed",
            {"terminal_non_completed": True},
            state=runtime_final_state,
            env_flag="LOG_A3_TRACE",
        )
        if failed_activity_payload:
            yield failed_activity_payload
        yield _stream_draft(
            "stream_error",
            {
                "error_type": "terminal_checkpoint_persist_failed",
                "message": "Authoritative final could not be persisted",
                "recoverable": False,
                "terminal_non_completed": True,
            },
        )
        finish_active_run(thread_id, {"run_status": RUN_STATUS_ERROR})
        return
    run_control_registry.clear_stop_signal(thread_id)
    for activity_payload in final_activity_payloads:
        yield activity_payload
    if qa_payload:
        emit_a3_trace(
            logger,
            "sse_qa_final",
            {"sent": True, **qa_final_trace_payload(qa_payload)},
            state=runtime_final_state,
            env_flag="LOG_A3_TRACE",
        )
        yield _stream_draft(
            "qa_final",
            {key: value for key, value in qa_payload.items() if key != "type"},
        )
    elif resource_payload:
        emit_a3_trace(
            logger,
            "sse_resource_final",
            {
                "sent": True,
                "schema_version": resource_payload.get("schema_version"),
                "resource_final_id": resource_payload.get(
                    "resource_final_id",
                    "",
                ),
                "payload_hash": resource_payload.get("payload_hash", ""),
                "resource_types": [
                    item.get("kind", "")
                    for item in (resource_payload.get("resources") or [])
                    if isinstance(item, dict)
                ],
                "resource_count": len(resource_payload.get("resources") or []),
                "blocked_count": len(resource_payload.get("blocked_resources") or []),
                "error_count": len(resource_payload.get("errors") or []),
                "terminal_status": resource_payload.get("terminal_status", ""),
                "validation": resource_payload.get("validation", {}),
                "summary_chars": len(str(resource_payload.get("summary") or "")),
            },
            state=runtime_final_state,
            env_flag="LOG_A3_TRACE",
        )
        yield _stream_draft(
            "resource_final",
            {key: value for key, value in resource_payload.items() if key != "type"},
        )
    elif recommendation_payload:
        emit_a3_trace(
            logger,
            "sse_recommendation_final",
            {
                "sent": True,
                "schema_version": recommendation_payload.get("schema_version"),
                "recommendation_final_id": recommendation_payload.get(
                    "recommendation_final_id",
                    "",
                ),
                "payload_hash": recommendation_payload.get("payload_hash", ""),
                "terminal_status": recommendation_payload.get(
                    "terminal_status",
                    "",
                ),
                "recommendation_count": len(
                    recommendation_payload.get("recommendations") or []
                ),
                "unavailable_reason": recommendation_payload.get("unavailable_reason"),
            },
            state=runtime_final_state,
            env_flag="LOG_A3_TRACE",
        )
        yield _stream_draft("recommendation_final", recommendation_payload)
    if completed_update_ok:
        finish_active_run(thread_id, {"run_status": RUN_STATUS_COMPLETED})


def _stream_context_payload(
    *,
    request_id: str,
    thread_id: str,
    graph_version: str,
) -> dict:
    payload = {
        "type": "stream_context",
        "schema_version": "stream_context_v1",
        "request_id": request_id,
        "thread_id": thread_id,
        "graph_version": graph_version,
    }
    # ``stream_context_v1`` has always required a graph version. Keep that
    # contract intact: a browser capability is additive only to a valid event.
    if graph_version:
        capability = current_frontend_performance_capability()
        if capability:
            payload["performance_telemetry"] = capability
    return payload


async def _generate_stream_drafts_impl(
    query: str,
    graph,
    thread_id: str,
    user_id: str | None = None,
    graph_version: str = "",
    *,
    request_id: str,
) -> AsyncGenerator[AgentStreamEventDraftV2, None]:
    """Produce native agent_stream_v2 drafts for one new graph request."""
    config = make_thread_config(thread_id)
    run_control_registry.clear_stop_signal(thread_id)

    # Inject profile context as a SystemMessage if a user profile exists
    messages = [HumanMessage(content=query)]
    profile_summary = ""
    if user_id:
        try:
            manager = get_profile_manager()
            profile_ctx = await manager.build_profile_context(user_id)
            if profile_ctx:
                messages.insert(0, SystemMessage(content=profile_ctx))
                profile_summary = sanitize_error_message(
                    profile_ctx,
                    max_chars=2000,
                )
                logger.info(
                    "Injected profile context user=%s (%d chars)",
                    user_id,
                    len(profile_ctx),
                )
        except Exception:
            logger.exception("Failed to load profile context user=%s", user_id)

    state_input = {
        "messages": messages,
        "request_id": request_id,
        "session_id": thread_id,
        "thread_id": thread_id,
        "context": CONTEXT_CLEAR,
        **initial_request_reset_transient_state(),
    }
    state_input["user_id"] = _explicit_user_id_state_value(user_id)
    if graph_version:
        state_input["graph_version"] = graph_version
    runtime_checkpointer_type = _graph_checkpointer_type(graph)
    state_input["runtime_capability_metadata"] = {
        "checkpointer_enabled": runtime_checkpointer_type != "none",
        "checkpointer_type": runtime_checkpointer_type,
    }
    if profile_summary:
        state_input["profile_summary"] = profile_summary
        state_input["learner_profile_summary"] = profile_summary
    _emit_graph_config_trace(graph, config, state_input)

    initial_request_context_window = {
        "current_request_id": request_id,
        "current_node": "",
        "last_event_count": 0,
    }
    initial_run_values = {
        "schema_version": RUN_CONTROL_SCHEMA_VERSION,
        "run_status": RUN_STATUS_RUNNING,
        "stop_requested": False,
        "stop_reason": "",
        "current_node": "",
        "last_completed_node": "",
        "resume_available": False,
        "stopped_at": "",
        "pending_interrupt_type": "",
        "profile_completion_request": {},
        "context_usage": {},
        "context_usage_history": [],
        "request_context_window": initial_request_context_window,
        "request_id": request_id,
        "session_id": thread_id,
        "thread_id": thread_id,
        "graph_version": graph_version,
    }
    try:
        await safe_update_thread_state(
            graph,
            config,
            initial_run_values,
            state=state_input,
            as_node="supervisor",
        )
    except Exception:
        logger.exception("Failed to initialize thread checkpoint thread=%s", thread_id)
        yield _stream_draft(
            "stream_error",
            {
                "error_type": "thread_checkpoint_initialization_failed",
                "message": "Thread checkpoint initialization failed",
                "recoverable": False,
            },
        )
        return

    run_state_snapshot = await graph.aget_state(config)
    snapshot_values = _state_values(run_state_snapshot)
    try:
        (
            compacted_values,
            compaction_payload,
        ) = await _prepare_full_compaction_for_new_request(
            graph,
            config,
            thread_id=thread_id,
            request_id=request_id,
            snapshot_values=snapshot_values,
            state_input=state_input,
        )
    except Exception as exc:
        emit_a3_trace(
            logger,
            "full_compaction.failed",
            {
                "error_type": type(exc).__name__,
                "recoverable": False,
            },
            state={
                "request_id": request_id,
                "thread_id": thread_id,
                "session_id": thread_id,
            },
            env_flag="LOG_A3_TRACE",
        )
        await _try_update_run_state(
            graph,
            config,
            {
                "run_status": RUN_STATUS_ERROR,
                "resume_available": False,
                "pending_interrupt_type": "",
            },
            state=state_input,
            persist_checkpoint=True,
        )
        yield _stream_draft(
            "stream_error",
            {
                "error_type": "full_compaction_failed",
                "message": "Full compaction failed",
                "recoverable": False,
                "thread_id": thread_id,
            },
        )
        return
    status_values = _new_request_status_values(
        compacted_values,
        initial_run_values,
    )
    request_context_window, thread_context_window = _context_window_status(
        status_values
    )
    start_active_run(
        thread_id,
        {
            "schema_version": RUN_CONTROL_SCHEMA_VERSION,
            "run_status": RUN_STATUS_RUNNING,
            "resume_available": False,
            "pending_interrupt_type": "",
            "profile_completion_request": {},
            "current_node": "",
            "last_completed_node": "",
            "request_context_window": request_context_window,
            "thread_context_window": thread_context_window,
            **_active_session_context_fields(status_values, thread_id=thread_id),
            "context_usage": {},
            "context_usage_history": status_values.get("context_usage_history")
            if isinstance(status_values.get("context_usage_history"), list)
            else [],
            "context_usage_report": status_values.get("context_usage_report")
            if isinstance(status_values.get("context_usage_report"), dict)
            else {},
            "context_usage_reports": status_values.get("context_usage_reports")
            if isinstance(status_values.get("context_usage_reports"), list)
            else [],
            "activity_timeline": status_values.get("activity_timeline")
            if isinstance(status_values.get("activity_timeline"), list)
            else [],
            "graph_version": graph_version,
            "llm_input_manifest": _last_llm_input_manifest(status_values),
            "llm_input_manifests": status_values.get("llm_input_manifests")
            if isinstance(status_values.get("llm_input_manifests"), list)
            else [],
            "thread_context_ledger": status_values.get("thread_context_ledger")
            if isinstance(status_values.get("thread_context_ledger"), dict)
            else {},
            "background_context_window": status_values.get("background_context_window")
            if isinstance(status_values.get("background_context_window"), dict)
            else {},
            "context_influence_ledger": status_values.get("context_influence_ledger")
            if isinstance(status_values.get("context_influence_ledger"), dict)
            else {},
            "last_resource_final_payload": status_values.get(
                "last_resource_final_payload"
            )
            if isinstance(status_values.get("last_resource_final_payload"), dict)
            else {},
            "last_recommendation_final_payload": _last_recommendation_final_payload(
                status_values,
                thread_id=thread_id,
            ),
            "last_provider_dispatch": status_values.get("last_provider_dispatch")
            if isinstance(status_values.get("last_provider_dispatch"), dict)
            else {},
            "compact_boundary": status_values.get("compact_boundary")
            if isinstance(status_values.get("compact_boundary"), dict)
            else {},
            "conversation_summary_v2": status_values.get("conversation_summary_v2")
            if isinstance(status_values.get("conversation_summary_v2"), dict)
            else {},
            "compaction_result": status_values.get("compaction_result")
            if isinstance(status_values.get("compaction_result"), dict)
            else {},
        },
    )

    yield _activity_update_draft("thread_id", {"thread_id": thread_id})
    stream_context_payload = _stream_context_payload(
        request_id=request_id,
        thread_id=thread_id,
        graph_version=graph_version,
    )
    if graph_version:
        yield _activity_update_draft(
            "stream_context",
            {
                key: value
                for key, value in stream_context_payload.items()
                if key != "type"
            },
        )
        graph_manifest_payload = graph_manifest_ref_payload(graph_version)
        yield _activity_update_draft(
            str(graph_manifest_payload.get("type") or "graph_manifest_ref"),
            {
                key: value
                for key, value in graph_manifest_payload.items()
                if key != "type"
            },
        )
    yield _activity_update_draft(
        "run_status",
        {
            "run_status": RUN_STATUS_RUNNING,
            "thread_id": thread_id,
        },
    )
    if compaction_payload:
        yield _activity_update_draft(
            "compaction_status",
            {
                "compaction_result": compaction_payload,
                "thread_context_window_v3": status_values.get(
                    "thread_context_window_v3",
                    {},
                ),
            },
        )

    # Record user input as episodic memory (non-fatal, fire-and-forget)
    if user_id:
        try:
            from src.memory.episodic import (
                compute_importance_for_user_query,
                write_episodic_memory,
            )

            importance, mem_type, content = compute_importance_for_user_query(
                query=query,
                subject="",
                resource_types=None,
            )
            await write_episodic_memory(
                {"user_id": user_id, "thread_id": thread_id},
                memory_type=mem_type,
                content=content,
                importance=importance,
            )
        except Exception:
            logger.exception("Failed to record user input episodic memory")

    async for chunk in _stream_graph_event_drafts(
        graph,
        state_input,
        config,
        thread_id,
        request_id=request_id,
        preserve_context_history=True,
    ):
        yield chunk

    # Record the conversation turn for profile evolution (non-fatal)
    if user_id:
        try:
            await manager.process_conversation(
                user_id=user_id,
                user_message=query,
                assistant_response="",
            )
            logger.debug("Profile turn recorded user=%s", user_id)
        except Exception:
            logger.exception("Profile recording failed (non-fatal) user=%s", user_id)


async def generate_stream_drafts(
    query: str,
    graph,
    thread_id: str | None = None,
    user_id: str | None = None,
    graph_version: str = "",
) -> AsyncGenerator[AgentStreamEventDraftV2, None]:
    """Produce one new request inside a complete performance root."""

    resolved_thread_id = thread_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    with observe_request_performance(
        request_id=request_id,
        thread_id=resolved_thread_id,
        user_id=user_id or "",
    ):
        async for chunk in _generate_stream_drafts_impl(
            query,
            graph,
            thread_id=resolved_thread_id,
            user_id=user_id,
            graph_version=graph_version,
            request_id=request_id,
        ):
            yield chunk


async def _new_request_stream_source(
    *,
    query: str,
    graph,
    thread_id: str,
    user_id: str | None,
    graph_version: str,
    request_id: str,
) -> AsyncGenerator[AgentStreamEventDraftV2, None]:
    with observe_request_performance(
        request_id=request_id,
        thread_id=thread_id,
        user_id=user_id or "",
    ):
        async for chunk in _generate_stream_drafts_impl(
            query,
            graph,
            thread_id=thread_id,
            user_id=user_id,
            graph_version=graph_version,
            request_id=request_id,
        ):
            yield chunk


def _stream_request_fingerprint(operation: str, payload: dict) -> str:
    """Hash request semantics without retaining raw request content in the journal."""

    encoded = json.dumps(
        {"operation": operation, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def _load_assessment_checkpoint_resources(
    graph,
    *,
    thread_id: str,
) -> AssessmentCheckpointResourcesV2:
    config = make_thread_config(thread_id)
    state_snapshot = await graph.aget_state(config)
    if not _has_checkpoint_state(state_snapshot):
        raise HTTPException(status_code=404, detail="assessment_checkpoint_not_found")
    values = _state_values(state_snapshot)
    raw = values.get("assessment_checkpoint_resources")
    if not isinstance(raw, dict) or not raw:
        raise HTTPException(
            status_code=404,
            detail="assessment_resources_not_found",
        )
    try:
        checkpoint = validate_assessment_checkpoint_resources_v2(raw)
    except AssessmentCheckpointError as exc:
        raise HTTPException(status_code=409, detail=exc.code) from exc
    if checkpoint.thread_id != thread_id:
        raise HTTPException(
            status_code=409,
            detail="assessment_checkpoint_thread_mismatch",
        )
    return checkpoint


def _build_assessment_attempt_service(
    graph,
    *,
    execution_lock: AssessmentExecutionLock,
) -> AssessmentAttemptService:
    async def load_journal(thread_id: str) -> object:
        config = make_thread_config(thread_id)
        state_snapshot = await graph.aget_state(config)
        if not _has_checkpoint_state(state_snapshot):
            raise RuntimeError("assessment checkpoint is unavailable")
        return _state_values(state_snapshot).get("assessment_attempt_journal", {})

    async def append_journal(
        thread_id: str,
        update: AssessmentAttemptJournalV1,
    ) -> None:
        config = make_thread_config(thread_id)
        state_snapshot = await graph.aget_state(config)
        if not _has_checkpoint_state(state_snapshot):
            raise RuntimeError("assessment checkpoint is unavailable")
        values = _state_values(state_snapshot)
        await safe_update_thread_state(
            graph,
            config,
            {"assessment_attempt_journal": update.model_dump(mode="json")},
            state=values,
        )

    return AssessmentAttemptService(
        idempotency=AssessmentCheckpointIdempotencyExecutor(
            load_journal=load_journal,
            append_journal=append_journal,
            execution_lock=execution_lock,
        ),
        error_classifier=classify_assessment_error_v1,
        adaptive_generator=generate_adaptive_practice_v1,
    )


class AssessmentHistoryProjectionError(RuntimeError):
    """Content-safe failure between durable assessment and history terminals."""

    def __init__(self, *, code: str) -> None:
        self.code = code
        super().__init__(f"{code}: assessment history projection failed")


async def _load_completed_assessment_record(
    graph,
    *,
    thread_id: str,
    request_id: str,
    expected_final,
) -> AssessmentAttemptRecordV1:
    """Reload the durable journal receipt used as history observed_at authority."""

    state_snapshot = await graph.aget_state(make_thread_config(thread_id))
    if not _has_checkpoint_state(state_snapshot):
        raise AssessmentHistoryProjectionError(
            code="assessment_history_checkpoint_unavailable"
        )
    raw_journal = _state_values(state_snapshot).get("assessment_attempt_journal")
    try:
        journal = validate_assessment_attempt_journal_v1(
            raw_journal,
            thread_id=thread_id,
        )
    except AssessmentAttemptJournalError as exc:
        raise AssessmentHistoryProjectionError(
            code="assessment_history_journal_invalid"
        ) from exc
    record = find_assessment_attempt_record_v1(
        journal,
        request_id=request_id,
    )
    if (
        record is None
        or record.status != "completed"
        or record.final is None
        or record.committed_at is None
    ):
        raise AssessmentHistoryProjectionError(
            code="assessment_history_completed_record_missing"
        )
    if record.final != expected_final:
        raise AssessmentHistoryProjectionError(code="assessment_history_final_mismatch")
    return record


def _assessment_stream_error_data(exc: Exception) -> dict:
    if isinstance(exc, AssessmentRequestConflict):
        return {
            "error_type": exc.code,
            "message": "Assessment request conflicts with an existing request",
            "recoverable": False,
        }
    if isinstance(exc, AssessmentIdentityError):
        return {
            "error_type": exc.code,
            "message": "Assessment target identity validation failed",
            "reason": exc.reason,
            "recoverable": False,
        }
    if isinstance(exc, AssessmentRecoveryRequired):
        return {
            "error_type": exc.code,
            "message": "Assessment request requires explicit recovery",
            "recoverable": False,
        }
    if isinstance(exc, AssessmentRecordedFailure):
        return {
            "error_type": exc.code,
            "message": (
                "Assessment attempt failed"
                if exc.stage == "assessment"
                else "Assessment dependency failed"
            ),
            "stage": exc.stage,
            "exception_type": sanitize_error_message(
                exc.exception_type,
                max_chars=120,
            ),
            "recoverable": False,
        }
    if isinstance(exc, AssessmentDependencyFailed):
        return {
            "error_type": exc.code,
            "message": "Assessment dependency failed",
            "stage": exc.stage,
            "exception_type": sanitize_error_message(
                exc.exception_type,
                max_chars=120,
            ),
            "recoverable": False,
        }
    if isinstance(exc, AssessmentAttemptServiceError):
        return {
            "error_type": exc.code,
            "message": "Assessment attempt failed",
            "recoverable": False,
        }
    if isinstance(exc, LearningGuidanceHistoryWriterError):
        return {
            "error_type": exc.code,
            "message": "Assessment history persistence failed",
            "recoverable": exc.code == "learning_guidance_history_persist_failed",
        }
    if isinstance(exc, AssessmentHistoryProjectionError):
        return {
            "error_type": exc.code,
            "message": "Assessment history projection failed",
            "recoverable": False,
        }
    return {
        "error_type": "assessment_attempt_failed",
        "message": "Assessment attempt failed",
        "recoverable": False,
    }


async def _assessment_attempt_stream_source(
    *,
    service: AssessmentAttemptService,
    thread_id: str,
    attempt: AssessmentAttemptV1,
    checkpoint: AssessmentCheckpointResourcesV2,
    graph,
    history_writer: LearningGuidanceHistoryWriterV1 | None,
) -> AsyncGenerator[AgentStreamEventDraftV2, None]:
    try:
        final = await service.submit(
            thread_id=thread_id,
            attempt=attempt,
            checkpoint=checkpoint,
        )
        resource = next(
            (
                item
                for item in checkpoint.resources
                if item.resource_id == final.resource_id
            ),
            None,
        )
        if resource is None:
            raise AssessmentHistoryProjectionError(
                code="assessment_history_resource_missing"
            )
        binding = resource.learning_guidance_binding
        if binding is None:
            yield _activity_update_draft(
                "assessment_history",
                {
                    "status": "unavailable",
                    "reason": "assessment_topic_binding_unavailable",
                    "request_id": final.request_id,
                    "resource_id": final.resource_id,
                    "question_id": final.question_id,
                },
            )
        else:
            if not isinstance(history_writer, LearningGuidanceHistoryWriterV1):
                raise AssessmentHistoryProjectionError(
                    code="assessment_history_writer_unavailable"
                )
            completed_record = await _load_completed_assessment_record(
                graph,
                thread_id=thread_id,
                request_id=attempt.request_id,
                expected_final=final,
            )
            write_result = await history_writer.write_assessment_once(
                binding=binding,
                record=completed_record,
            )
            yield _activity_update_draft(
                "assessment_history",
                {
                    "status": write_result.status,
                    "history_id": write_result.history_id,
                    "request_id": final.request_id,
                    "resource_id": final.resource_id,
                    "question_id": final.question_id,
                },
            )
    except Exception as exc:
        yield _stream_draft("stream_error", _assessment_stream_error_data(exc))
        return
    yield _stream_draft("assessment_final", final.model_dump(mode="json"))


def _new_thread_id_for_request(request_id: str) -> str:
    """Keep retries without a client thread_id bound to one stable thread."""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"a3-study-agent:request:{request_id}"))


def _explicit_user_id_state_value(user_id: str | None) -> str:
    """Preserve only the request user id; never derive identity from thread state."""

    return user_id if user_id is not None else ""


async def _generate_resume_stream_drafts_impl(
    edited_plan: str,
    feedback: str | None,
    graph,
    thread_id: str,
    memory_use_choice: str | None = None,
    profile_completion: dict | None = None,
    graph_version: str = "",
    *,
    resume_request_id: str,
) -> AsyncGenerator[AgentStreamEventDraftV2, None]:
    """Resume an interrupted graph and produce remaining native drafts.

    Args:
        edited_plan: The user-edited plan text to resume with.
        feedback: Optional feedback text for AI-driven plan revision.
        graph: The compiled LangGraph instance from app.state.
        thread_id: Session ID identifying the interrupted graph state.
    """
    config = make_thread_config(thread_id)
    state_snapshot = await graph.aget_state(config)
    pending_type = _pending_interrupt_type(state_snapshot)
    if pending_type == "user_stop":
        yield _stream_draft(
            "stopped",
            {
                "run_status": RUN_STATUS_STOPPED,
                "thread_id": thread_id,
                "resume_available": True,
                "pending_interrupt_type": "user_stop",
                "message": "use /threads/{thread_id}/continue for user_stop interrupts",
            },
        )
        return
    if profile_completion is not None and pending_type != "profile_completion_required":
        payload = {
            "type": "error",
            "error_type": "profile_completion_checkpoint_missing",
            "message": "profile_completion_checkpoint_missing",
            "thread_id": thread_id,
            "pending_interrupt_type": pending_type,
            "resume_available": False,
            "recoverable": False,
        }
        emit_a3_trace(
            logger,
            "profile_completion.resume_failed",
            {
                "thread_id": thread_id,
                "pending_interrupt_type": pending_type,
                "error_type": "profile_completion_checkpoint_missing",
            },
            state={"thread_id": thread_id, "session_id": thread_id},
            env_flag="LOG_A3_TRACE",
        )
        finish_active_run(
            thread_id,
            {
                "run_status": RUN_STATUS_ERROR,
                "resume_available": False,
                "pending_interrupt_type": "",
                "error_type": "profile_completion_checkpoint_missing",
            },
        )
        yield _stream_draft(
            "stream_error",
            {key: value for key, value in payload.items() if key != "type"},
        )
        return

    resume_value: object
    if memory_use_choice:
        resume_value = {"type": "memory_confirmation", "choice": memory_use_choice}
    elif profile_completion is not None:
        resume_value = {
            "type": "profile_completion_required",
            "profile_completion": profile_completion,
        }
    elif feedback:
        resume_value = {"action": "feedback", "text": feedback}
    else:
        resume_value = edited_plan

    resume_input = Command(
        resume=resume_value,
        update={"request_id": resume_request_id, "thread_id": thread_id},
    )
    _emit_graph_config_trace(
        graph,
        config,
        {
            "request_id": resume_request_id,
            "session_id": thread_id,
            "thread_id": thread_id,
        },
    )
    status_values = _state_values(state_snapshot)
    request_context_window, thread_context_window = _context_window_status(
        status_values
    )
    if not request_context_window.get("current_request_id"):
        request_context_window["current_request_id"] = resume_request_id
    start_active_run(
        thread_id,
        {
            "schema_version": RUN_CONTROL_SCHEMA_VERSION,
            "run_status": RUN_STATUS_CONTINUING,
            "resume_available": False,
            "pending_interrupt_type": "",
            "profile_completion_request": {},
            "current_node": "",
            "last_completed_node": "",
            "request_context_window": request_context_window,
            "thread_context_window": thread_context_window,
            **_active_session_context_fields(status_values, thread_id=thread_id),
            "context_usage": status_values.get("context_usage")
            if isinstance(status_values.get("context_usage"), dict)
            else {},
            "context_usage_history": status_values.get("context_usage_history")
            if isinstance(status_values.get("context_usage_history"), list)
            else [],
            "context_usage_report": status_values.get("context_usage_report")
            if isinstance(status_values.get("context_usage_report"), dict)
            else {},
            "context_usage_reports": status_values.get("context_usage_reports")
            if isinstance(status_values.get("context_usage_reports"), list)
            else [],
            "activity_timeline": status_values.get("activity_timeline")
            if isinstance(status_values.get("activity_timeline"), list)
            else [],
            "graph_version": graph_version,
            "llm_input_manifest": _last_llm_input_manifest(status_values),
            "llm_input_manifests": status_values.get("llm_input_manifests")
            if isinstance(status_values.get("llm_input_manifests"), list)
            else [],
            "thread_context_ledger": status_values.get("thread_context_ledger")
            if isinstance(status_values.get("thread_context_ledger"), dict)
            else {},
            "background_context_window": status_values.get("background_context_window")
            if isinstance(status_values.get("background_context_window"), dict)
            else {},
            "context_influence_ledger": status_values.get("context_influence_ledger")
            if isinstance(status_values.get("context_influence_ledger"), dict)
            else {},
            "last_resource_final_payload": status_values.get(
                "last_resource_final_payload"
            )
            if isinstance(status_values.get("last_resource_final_payload"), dict)
            else {},
            "last_recommendation_final_payload": _last_recommendation_final_payload(
                status_values,
                thread_id=thread_id,
            ),
        },
    )

    stream_context_payload = _stream_context_payload(
        request_id=resume_request_id,
        thread_id=thread_id,
        graph_version=graph_version,
    )
    if graph_version:
        yield _activity_update_draft(
            "stream_context",
            {
                key: value
                for key, value in stream_context_payload.items()
                if key != "type"
            },
        )
        graph_manifest_payload = graph_manifest_ref_payload(graph_version)
        yield _activity_update_draft(
            str(graph_manifest_payload.get("type") or "graph_manifest_ref"),
            {
                key: value
                for key, value in graph_manifest_payload.items()
                if key != "type"
            },
        )
    yield _activity_update_draft(
        "run_status",
        {
            "run_status": RUN_STATUS_CONTINUING,
            "thread_id": thread_id,
        },
    )

    async for chunk in _stream_graph_event_drafts(
        graph,
        resume_input,
        config,
        thread_id,
        request_id=resume_request_id,
        preserve_context_history=True,
    ):
        yield chunk


async def generate_resume_stream_drafts(
    edited_plan: str,
    feedback: str | None,
    graph,
    thread_id: str,
    memory_use_choice: str | None = None,
    profile_completion: dict | None = None,
    graph_version: str = "",
) -> AsyncGenerator[AgentStreamEventDraftV2, None]:
    """Resume one request inside a complete request performance root."""

    resume_request_id = str(uuid.uuid4())
    with observe_request_performance(
        request_id=resume_request_id,
        thread_id=thread_id,
    ):
        async for chunk in _generate_resume_stream_drafts_impl(
            edited_plan,
            feedback,
            graph,
            thread_id,
            memory_use_choice=memory_use_choice,
            profile_completion=profile_completion,
            graph_version=graph_version,
            resume_request_id=resume_request_id,
        ):
            yield chunk


async def _resume_stream_source(
    *,
    edited_plan: str,
    feedback: str | None,
    graph,
    thread_id: str,
    memory_use_choice: str | None,
    profile_completion: dict | None,
    graph_version: str,
    request_id: str,
) -> AsyncGenerator[AgentStreamEventDraftV2, None]:
    with observe_request_performance(request_id=request_id, thread_id=thread_id):
        async for chunk in _generate_resume_stream_drafts_impl(
            edited_plan,
            feedback,
            graph,
            thread_id,
            memory_use_choice=memory_use_choice,
            profile_completion=profile_completion,
            graph_version=graph_version,
            resume_request_id=request_id,
        ):
            yield chunk


async def get_thread_status_payload(graph, thread_id: str) -> ThreadStatusResponse:
    active_run = get_active_run(thread_id)
    if active_run is not None:
        return _thread_status_from_active_run(thread_id, active_run)
    config = make_thread_config(thread_id)
    state_snapshot = await graph.aget_state(config)
    if not _has_checkpoint_state(state_snapshot):
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return _thread_status_from_snapshot(thread_id, state_snapshot)


async def request_thread_stop(graph, thread_id: str, reason: str) -> dict:
    config = make_thread_config(thread_id)
    signal = run_control_registry.request_stop(thread_id, reason or "user_stop")
    values = {
        "schema_version": RUN_CONTROL_SCHEMA_VERSION,
        "run_status": RUN_STATUS_STOPPING,
        "stop_requested": True,
        "stop_reason": signal.reason,
        "stop_requested_at": signal.requested_at,
        "resume_available": False,
    }
    await _update_run_state(graph, config, values)
    emit_a3_trace(
        logger,
        "run_stop_requested",
        {
            "thread_id": thread_id,
            "requested_at": signal.requested_at,
            "reason": signal.reason,
        },
        state={"thread_id": thread_id, "session_id": thread_id},
        env_flag="LOG_A3_TRACE",
    )
    return {
        "ok": True,
        "thread_id": thread_id,
        "run_status": RUN_STATUS_STOPPING,
        "stop_requested": True,
        "requested_at": signal.requested_at,
    }


async def _generate_continue_stream_drafts_impl(
    graph,
    thread_id: str,
    *,
    graph_version: str = "",
    continue_request_id: str,
) -> AsyncGenerator[AgentStreamEventDraftV2, None]:
    config = make_thread_config(thread_id)
    state_snapshot = await graph.aget_state(config)
    if not _has_checkpoint_state(state_snapshot):
        yield _stream_draft(
            "stream_error",
            {
                "error_type": "not_resumable",
                "run_status": RUN_STATUS_NOT_RESUMABLE,
                "thread_id": thread_id,
                "resume_available": False,
                "message": "checkpoint_not_found",
                "recoverable": False,
            },
        )
        return

    pending_type = _pending_interrupt_type(state_snapshot)
    if pending_type in {"plan_review", "memory_confirmation"}:
        yield _stream_draft(
            "stream_error",
            {
                "error_type": "not_resumable",
                "run_status": RUN_STATUS_NOT_RESUMABLE,
                "thread_id": thread_id,
                "resume_available": False,
                "pending_interrupt_type": pending_type,
                "message": "pending HIL interrupt must be resumed with /resume",
                "recoverable": True,
            },
        )
        return
    if pending_type != "user_stop":
        yield _stream_draft(
            "stream_error",
            {
                "error_type": "not_resumable",
                "run_status": RUN_STATUS_NOT_RESUMABLE,
                "thread_id": thread_id,
                "resume_available": False,
                "pending_interrupt_type": pending_type,
                "message": "no pending user_stop interrupt",
                "recoverable": False,
            },
        )
        return

    run_control_registry.clear_stop_signal(thread_id)
    status_values = _state_values(state_snapshot)
    request_context_window, thread_context_window = _context_window_status(
        status_values
    )
    start_active_run(
        thread_id,
        {
            "schema_version": RUN_CONTROL_SCHEMA_VERSION,
            "run_status": RUN_STATUS_CONTINUING,
            "resume_available": False,
            "pending_interrupt_type": "",
            "current_node": "",
            "last_completed_node": "",
            "request_context_window": request_context_window,
            "thread_context_window": thread_context_window,
            **_active_session_context_fields(status_values, thread_id=thread_id),
            "context_usage": status_values.get("context_usage")
            if isinstance(status_values.get("context_usage"), dict)
            else {},
            "context_usage_history": status_values.get("context_usage_history")
            if isinstance(status_values.get("context_usage_history"), list)
            else [],
            "context_usage_report": status_values.get("context_usage_report")
            if isinstance(status_values.get("context_usage_report"), dict)
            else {},
            "context_usage_reports": status_values.get("context_usage_reports")
            if isinstance(status_values.get("context_usage_reports"), list)
            else [],
            "activity_timeline": status_values.get("activity_timeline")
            if isinstance(status_values.get("activity_timeline"), list)
            else [],
            "graph_version": graph_version,
            "llm_input_manifest": _last_llm_input_manifest(status_values),
            "llm_input_manifests": status_values.get("llm_input_manifests")
            if isinstance(status_values.get("llm_input_manifests"), list)
            else [],
            "thread_context_ledger": status_values.get("thread_context_ledger")
            if isinstance(status_values.get("thread_context_ledger"), dict)
            else {},
            "background_context_window": status_values.get("background_context_window")
            if isinstance(status_values.get("background_context_window"), dict)
            else {},
            "context_influence_ledger": status_values.get("context_influence_ledger")
            if isinstance(status_values.get("context_influence_ledger"), dict)
            else {},
            "last_resource_final_payload": status_values.get(
                "last_resource_final_payload"
            )
            if isinstance(status_values.get("last_resource_final_payload"), dict)
            else {},
            "last_recommendation_final_payload": _last_recommendation_final_payload(
                status_values,
                thread_id=thread_id,
            ),
        },
    )
    await _update_run_state(
        graph,
        config,
        {
            "run_status": RUN_STATUS_CONTINUING,
            "stop_requested": False,
            "stop_reason": "",
            "stop_requested_at": "",
            "resume_available": False,
            "pending_interrupt_type": "",
        },
    )
    emit_a3_trace(
        logger,
        "run_continue_requested",
        {
            "thread_id": thread_id,
            "pending_interrupt_type": "user_stop",
        },
        state={"thread_id": thread_id, "session_id": thread_id},
        env_flag="LOG_A3_TRACE",
    )
    stream_context_payload = _stream_context_payload(
        request_id=continue_request_id,
        thread_id=thread_id,
        graph_version=graph_version,
    )
    if graph_version:
        yield _activity_update_draft(
            "stream_context",
            {
                key: value
                for key, value in stream_context_payload.items()
                if key != "type"
            },
        )
        graph_manifest_payload = graph_manifest_ref_payload(graph_version)
        yield _activity_update_draft(
            str(graph_manifest_payload.get("type") or "graph_manifest_ref"),
            {
                key: value
                for key, value in graph_manifest_payload.items()
                if key != "type"
            },
        )
    yield _activity_update_draft(
        "run_status",
        {
            "run_status": RUN_STATUS_CONTINUING,
            "thread_id": thread_id,
        },
    )

    resume_input = Command(
        resume={"type": "user_stop", "action": "continue"},
        update={"request_id": continue_request_id, "thread_id": thread_id},
    )
    _emit_graph_config_trace(
        graph,
        config,
        {
            "request_id": continue_request_id,
            "session_id": thread_id,
            "thread_id": thread_id,
        },
    )
    async for chunk in _stream_graph_event_drafts(
        graph,
        resume_input,
        config,
        thread_id,
        request_id=continue_request_id,
        preserve_context_history=True,
    ):
        yield chunk


async def generate_continue_stream_drafts(
    graph,
    thread_id: str,
    *,
    graph_version: str = "",
) -> AsyncGenerator[AgentStreamEventDraftV2, None]:
    """Continue one stopped request inside a complete request performance root."""

    continue_request_id = str(uuid.uuid4())
    with observe_request_performance(
        request_id=continue_request_id,
        thread_id=thread_id,
    ):
        async for chunk in _generate_continue_stream_drafts_impl(
            graph,
            thread_id,
            graph_version=graph_version,
            continue_request_id=continue_request_id,
        ):
            yield chunk


async def _continue_stream_source(
    *,
    graph,
    thread_id: str,
    graph_version: str,
    request_id: str,
) -> AsyncGenerator[AgentStreamEventDraftV2, None]:
    with observe_request_performance(request_id=request_id, thread_id=thread_id):
        async for chunk in _generate_continue_stream_drafts_impl(
            graph,
            thread_id,
            graph_version=graph_version,
            continue_request_id=request_id,
        ):
            yield chunk


def _app_graph_version(fastapi_app: FastAPI) -> str:
    value = getattr(fastapi_app.state, "graph_version", "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    error = getattr(fastapi_app.state, "graph_manifest_error", None)
    detail = (
        error
        if isinstance(error, dict)
        else {
            "schema_version": "graph_manifest_error_v1",
            "error": "graph_manifest_unavailable",
            "reason": "graph manifest is unavailable",
            "error_type": "GraphManifestBuildError",
        }
    )
    raise HTTPException(status_code=503, detail=detail)


@app.get("/graph/manifest", response_model=GraphManifest)
async def graph_manifest_endpoint(request: Request):
    manifest = getattr(request.app.state, "graph_manifest", None)
    if isinstance(manifest, GraphManifest):
        emit_a3_trace(
            logger,
            "graph_manifest.served",
            graph_manifest_status_payload(manifest),
            state={},
            env_flag="LOG_A3_TRACE",
        )
        return manifest
    error = getattr(request.app.state, "graph_manifest_error", None)
    detail = (
        error
        if isinstance(error, dict)
        else {
            "schema_version": "graph_manifest_error_v1",
            "error": "graph_manifest_unavailable",
            "reason": "graph manifest cache is unavailable",
            "error_type": "GraphManifestBuildError",
        }
    )
    raise HTTPException(status_code=503, detail=detail)


@app.post(FRONTEND_PERFORMANCE_ENDPOINT_PATH, status_code=204)
async def frontend_performance_endpoint(request: Request):
    """Accept one authenticated, content-free browser milestone batch."""

    service = get_performance_service()
    frontend_config = service.config.frontend_ingestion
    if not frontend_config.enabled:
        raise HTTPException(status_code=503, detail="frontend_performance_disabled")
    raw = await request.body()
    if len(raw) > frontend_config.max_payload_bytes:
        raise HTTPException(
            status_code=413,
            detail="frontend_performance_payload_too_large",
        )
    try:
        payload = FrontendPerformanceBatchV1.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        emit_a3_trace(
            logger,
            "performance.frontend.batch.rejected",
            {
                "reason_code": "frontend_performance_payload_invalid",
                "error_type": type(exc).__name__,
                "payload_bytes": len(raw),
            },
            state={},
            env_flag="LOG_PERFORMANCE_TRACE",
            level="info",
        )
        raise HTTPException(
            status_code=422,
            detail="frontend_performance_payload_invalid",
        ) from exc
    try:
        service.accept_frontend_batch(
            authorization=request.headers.get("authorization", ""),
            origin=request.headers.get("origin", ""),
            raw_size=len(raw),
            payload=payload,
        )
    except FrontendPerformanceRejected as exc:
        emit_a3_trace(
            logger,
            "performance.frontend.batch.rejected",
            {
                "reason_code": exc.code,
                "status_code": exc.status_code,
                "payload_bytes": len(raw),
            },
            state={
                "request_id": payload.request_id,
                "thread_id": payload.thread_id,
            },
            env_flag="LOG_PERFORMANCE_TRACE",
            level="info",
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
    return Response(status_code=204)


@app.post(
    "/stream",
    response_class=AgentEventStreamResponse,
    responses=_SSE_OPENAPI_RESPONSES,
)
async def stream_endpoint(chat: ChatRequest, request: Request):
    graph_version = _app_graph_version(request.app)
    request_id = str(chat.request_id)
    if not chat.query.strip():
        raise HTTPException(status_code=422, detail="query_must_not_be_blank")
    if chat.thread_id is not None and not chat.thread_id.strip():
        raise HTTPException(status_code=422, detail="thread_id_must_not_be_blank")
    thread_id = chat.thread_id or _new_thread_id_for_request(request_id)
    operation = "new_request"
    request_fingerprint = _stream_request_fingerprint(
        operation,
        {"query": chat.query, "user_id": chat.user_id},
    )
    stream_id = str(uuid.uuid4())
    try:
        session = await stream_session_manager.create(
            stream_id=stream_id,
            request_id=request_id,
            thread_id=thread_id,
            operation=operation,
            request_fingerprint=request_fingerprint,
            source=_new_request_stream_source(
                query=chat.query,
                graph=request.app.state.graph,
                thread_id=thread_id,
                user_id=chat.user_id,
                graph_version=graph_version,
                request_id=request_id,
            ),
        )
        response_body = session.subscribe(after_sequence=0)
    except (StreamSessionExpiredError, StreamJournalExpiredError) as exc:
        raise HTTPException(status_code=410, detail="stream_request_expired") from exc
    except StreamSessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return AgentEventStreamResponse(
        response_body,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post(
    "/resume",
    response_class=AgentEventStreamResponse,
    responses=_SSE_OPENAPI_RESPONSES,
)
async def resume_endpoint(req: ResumeRequest, request: Request):
    graph_version = _app_graph_version(request.app)
    if not req.thread_id.strip():
        raise HTTPException(status_code=422, detail="thread_id_must_not_be_blank")
    profile_completion = (
        req.profile_completion.model_dump()
        if req.profile_completion is not None
        else None
    )
    request_id = str(req.request_id)
    operation = "resume"
    request_fingerprint = _stream_request_fingerprint(
        operation,
        {
            "edited_plan": req.edited_plan,
            "feedback": req.feedback,
            "memory_use_choice": req.memory_use_choice,
            "profile_completion": profile_completion,
        },
    )
    stream_id = str(uuid.uuid4())
    try:
        session = await stream_session_manager.create(
            stream_id=stream_id,
            request_id=request_id,
            thread_id=req.thread_id,
            operation=operation,
            request_fingerprint=request_fingerprint,
            source=_resume_stream_source(
                edited_plan=req.edited_plan,
                feedback=req.feedback,
                graph=request.app.state.graph,
                thread_id=req.thread_id,
                memory_use_choice=req.memory_use_choice,
                profile_completion=profile_completion,
                graph_version=graph_version,
                request_id=request_id,
            ),
        )
        response_body = session.subscribe(after_sequence=0)
    except (StreamSessionExpiredError, StreamJournalExpiredError) as exc:
        raise HTTPException(status_code=410, detail="stream_request_expired") from exc
    except StreamSessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return AgentEventStreamResponse(
        response_body,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/threads/{thread_id}/stop")
async def stop_thread_endpoint(thread_id: str, req: StopRequest, request: Request):
    return await request_thread_stop(request.app.state.graph, thread_id, req.reason)


@app.post(
    "/threads/{thread_id}/assessment-attempts",
    response_class=AgentEventStreamResponse,
    responses=_SSE_OPENAPI_RESPONSES,
)
async def assessment_attempt_endpoint(
    thread_id: str,
    attempt: AssessmentAttemptV1,
    request: Request,
):
    if not thread_id.strip():
        raise HTTPException(status_code=422, detail="thread_id_must_not_be_blank")
    service = getattr(request.app.state, "assessment_attempt_service", None)
    if not isinstance(service, AssessmentAttemptService):
        raise HTTPException(
            status_code=503,
            detail="assessment_service_unavailable",
        )
    checkpoint = await _load_assessment_checkpoint_resources(
        request.app.state.graph,
        thread_id=thread_id,
    )
    request_id = attempt.request_id
    operation = "assessment_attempt"
    request_fingerprint = stable_assessment_attempt_hash(
        thread_id=thread_id,
        attempt=attempt,
    )
    stream_id = str(uuid.uuid4())
    try:
        session = await stream_session_manager.create(
            stream_id=stream_id,
            request_id=request_id,
            thread_id=thread_id,
            operation=operation,
            request_fingerprint=request_fingerprint,
            source=_assessment_attempt_stream_source(
                service=service,
                thread_id=thread_id,
                attempt=attempt,
                checkpoint=checkpoint,
                graph=request.app.state.graph,
                history_writer=getattr(
                    request.app.state,
                    "learning_guidance_history_writer",
                    None,
                ),
            ),
            allow_recoverable_retry=True,
        )
        response_body = session.subscribe(after_sequence=0)
    except (StreamSessionExpiredError, StreamJournalExpiredError) as exc:
        raise HTTPException(status_code=410, detail="stream_request_expired") from exc
    except StreamSessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return AgentEventStreamResponse(
        response_body,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/threads/{thread_id}/status", response_model=ThreadStatusResponse)
async def thread_status_endpoint(thread_id: str, request: Request):
    return await get_thread_status_payload(request.app.state.graph, thread_id)


@app.post(
    "/threads/{thread_id}/continue",
    response_class=AgentEventStreamResponse,
    responses=_SSE_OPENAPI_RESPONSES,
)
async def continue_thread_endpoint(
    thread_id: str,
    req: ContinueRequest,
    request: Request,
):
    graph_version = _app_graph_version(request.app)
    if not thread_id.strip():
        raise HTTPException(status_code=422, detail="thread_id_must_not_be_blank")
    request_id = str(req.request_id)
    operation = "continue"
    request_fingerprint = _stream_request_fingerprint(operation, {})
    stream_id = str(uuid.uuid4())
    try:
        session = await stream_session_manager.create(
            stream_id=stream_id,
            request_id=request_id,
            thread_id=thread_id,
            operation=operation,
            request_fingerprint=request_fingerprint,
            source=_continue_stream_source(
                graph=request.app.state.graph,
                thread_id=thread_id,
                graph_version=graph_version,
                request_id=request_id,
            ),
        )
        response_body = session.subscribe(after_sequence=0)
    except (StreamSessionExpiredError, StreamJournalExpiredError) as exc:
        raise HTTPException(status_code=410, detail="stream_request_expired") from exc
    except StreamSessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return AgentEventStreamResponse(
        response_body,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get(
    "/streams/{stream_id}",
    response_class=AgentEventStreamResponse,
    responses=_SSE_OPENAPI_RESPONSES,
)
async def reconnect_stream_endpoint(stream_id: str, request: Request):
    try:
        session = await stream_session_manager.get(stream_id)
        after_sequence = parse_last_event_id(
            request.headers.get("last-event-id", ""),
            expected_stream_id=stream_id,
        )
        response_body = session.subscribe(after_sequence=after_sequence)
    except (StreamSessionExpiredError, StreamJournalExpiredError) as exc:
        raise HTTPException(status_code=410, detail="stream_session_expired") from exc
    except StreamSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="stream_session_not_found") from exc
    except StreamContractError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return AgentEventStreamResponse(
        response_body,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/dev/threads/{thread_id}/memory/clear")
async def clear_thread_memory_endpoint(thread_id: str, request: Request):
    return await clear_persistent_memory_for_thread(request.app.state.graph, thread_id)


@app.get("/artifacts/mindmaps/{artifact_id}/{filename}")
async def download_mindmap_artifact(artifact_id: str, filename: str):
    root = get_mindmap_artifact_dir()
    artifact_path = (root / artifact_id / filename).resolve()
    try:
        artifact_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Artifact not found")

    if not artifact_path.is_file() or artifact_path.suffix.lower() != ".xmind":
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(
        artifact_path,
        media_type="application/vnd.xmind.workbook",
        filename=filename,
    )


@app.get("/artifacts/review-docs/{artifact_id}/{filename}")
async def download_review_doc_artifact(artifact_id: str, filename: str):
    root = get_review_doc_artifact_dir()
    artifact_path = (root / artifact_id / filename).resolve()
    try:
        artifact_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Artifact not found")

    if not artifact_path.is_file() or artifact_path.suffix.lower() not in {
        ".md",
        ".docx",
    }:
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if artifact_path.suffix.lower() == ".docx"
        else "text/markdown; charset=utf-8"
    )

    return FileResponse(
        artifact_path,
        media_type=media_type,
        filename=filename,
    )


@app.get("/artifacts/exercises/{artifact_id}/{filename}")
async def download_exercise_artifact(artifact_id: str, filename: str):
    root = get_exercise_artifact_dir()
    artifact_path = (root / artifact_id / filename).resolve()
    try:
        artifact_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Artifact not found")

    if not artifact_path.is_file() or artifact_path.suffix.lower() not in {
        ".md",
        ".docx",
    }:
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if artifact_path.suffix.lower() == ".docx"
        else "text/markdown; charset=utf-8"
    )

    return FileResponse(
        artifact_path,
        media_type=media_type,
        filename=filename,
    )


@app.get("/artifacts/code-practice/{artifact_id}/{filename}")
async def download_code_practice_artifact(artifact_id: str, filename: str):
    root = get_code_practice_artifact_dir()
    artifact_path = (root / artifact_id / filename).resolve()
    try:
        artifact_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Artifact not found")

    suffix = artifact_path.suffix.lower()
    if not artifact_path.is_file() or suffix not in {".md", ".docx", ".py"}:
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type = {
        ".md": "text/markdown; charset=utf-8",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".py": "text/x-python",
    }[suffix]

    return FileResponse(
        artifact_path,
        media_type=media_type,
        filename=filename,
    )


@app.get("/artifacts/video-scripts/{artifact_id}/{filename}")
async def download_video_script_artifact(artifact_id: str, filename: str):
    root = get_video_script_artifact_dir()
    decoded_filename = unquote(filename)
    artifact_path = (root / artifact_id / decoded_filename).resolve()
    try:
        artifact_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Artifact not found")

    suffix = artifact_path.suffix.lower()
    if not artifact_path.is_file() or suffix not in {".md", ".docx", ".srt"}:
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type = {
        ".md": "text/markdown; charset=utf-8",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".srt": "application/x-subrip; charset=utf-8",
    }[suffix]

    return FileResponse(
        artifact_path,
        media_type=media_type,
        filename=decoded_filename,
    )


@app.get("/artifacts/video-animations/{artifact_id}/{filename}")
async def download_video_animation_artifact(artifact_id: str, filename: str):
    root = get_video_animation_artifact_dir()
    decoded_filename = unquote(filename)
    artifact_path = (root / artifact_id / decoded_filename).resolve()
    try:
        artifact_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Artifact not found")

    suffix = artifact_path.suffix.lower()
    if not artifact_path.is_file() or suffix not in {".html", ".json", ".srt", ".mp4"}:
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type = {
        ".html": "text/html; charset=utf-8",
        ".json": "application/json",
        ".srt": "application/x-subrip; charset=utf-8",
        ".mp4": "video/mp4",
    }[suffix]

    return FileResponse(
        artifact_path,
        media_type=media_type,
        filename=decoded_filename,
    )


# User profile and onboarding


def _onboarding_base_profile(
    req: CompiledOnboardRequestV2,
    *,
    now: str,
) -> UserProfile:
    """Build ordinary profile data from the same explicit topic self-report."""

    skills = {
        item.topic_id: SkillEntry(
            level=item.level,
            confidence=item.confidence,
            last_observed=now,
            evidence_count=1,
        )
        for item in req.profile.skills
    }
    goals = [
        Goal(
            goal=item.goal,
            importance=item.importance,
            progress=item.progress,
            created_at=now,
        )
        for item in req.profile.goals
    ]
    learning_style = LearningStyle()
    for item in req.profile.preferences:
        setattr(learning_style, item.dimension, item.strength)
    subjects = tuple(dict.fromkeys(item.subject for item in req.profile.skills))
    observations = [
        AgentObservation(
            content=f"用户自述年级: {req.grade}",
            category="general",
            importance=0.8,
            created_at=now,
        ),
        AgentObservation(
            content=(
                f"用户显式选择 {len(req.profile.skills)} 个 KG 学习主题，"
                f"科目: {', '.join(subjects)}"
            ),
            category="goal",
            importance=0.9,
            created_at=now,
        ),
    ]
    return UserProfile(
        user_id=req.profile.user_id,
        skills=skills,
        learning_style=learning_style,
        goals=goals,
        dislikes=list(req.dislikes),
        agent_observations=observations,
        extra={
            "nickname": req.nickname,
            "grade": req.grade,
            "onboarding_completed": True,
        },
        created_at=now,
        updated_at=now,
    )


@app.post("/onboard", response_model=OnboardResultV2)
async def onboard_endpoint(req: OnboardRequest, request: Request):
    """Create or replay one strict topic-bound onboarding profile."""

    compiled = compile_onboard_request_v2(req)
    writer = getattr(
        request.app.state,
        "learning_guidance_profile_writer",
        None,
    )
    if not isinstance(writer, LearningGuidanceProfileWriterV1):
        raise HTTPException(
            status_code=503,
            detail="learning_guidance_profile_writer_unavailable",
        )
    now = datetime.now(timezone.utc).isoformat()
    try:
        result = await writer.create_once(
            req.profile,
            base_profile=_onboarding_base_profile(compiled, now=now),
            source=compiled.profile_write_source,
        )
    except ProfileWriterError as exc:
        status_code = 409 if exc.code == "profile_write_request_conflict" else 422
        raise HTTPException(status_code=status_code, detail=exc.code) from exc

    logger.info(
        "Topic-bound onboarding profile %s user=%s skills=%d goals=%d",
        result.status,
        compiled.profile.user_id,
        len(result.binding.skills),
        len(result.binding.goals),
    )
    return OnboardResultV2(
        schema_version="onboard_result_v2",
        status=result.status,
        request_id=result.request_id,
        user_id=compiled.profile.user_id,
        summary=result.profile.to_summary(),
        skills_count=len(result.binding.skills),
        goals_count=len(result.binding.goals),
        preferences_count=len(result.binding.preferences),
    )


@app.get("/learning-guidance/catalog", response_model=LearningGuidanceCatalogV1)
async def learning_guidance_catalog_endpoint(request: Request):
    """Expose the exact curated subject/topic identities accepted by onboarding."""

    runtime = getattr(request.app.state, "learning_guidance_runtime", None)
    if not isinstance(runtime, LearningGuidanceRuntime):
        raise HTTPException(
            status_code=503,
            detail="learning_guidance_runtime_unavailable",
        )
    graph = runtime.knowledge_graph
    return LearningGuidanceCatalogV1.model_validate(
        {
            "schema_version": "learning_guidance_catalog_v1",
            "data_version": graph.data_version,
            "artifact_fingerprint": graph.artifact_fingerprint,
            "subjects": [
                {
                    "subject_id": subject.subject_id,
                    "title": subject.title,
                    "topics": [
                        {
                            "topic_id": topic.topic_id,
                            "title": topic.title,
                        }
                        for topic in subject.topics
                    ],
                }
                for subject in graph.subjects
            ],
        },
        strict=True,
    )


@app.get("/profile/{user_id}")
async def get_profile_endpoint(user_id: str):
    """Return the current user profile, or 404 if not found."""
    manager = get_profile_manager()
    profile = await manager.store.load(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {
        "user_id": user_id,
        "has_profile": True,
        "summary": profile.to_summary(),
        "skills": {
            name: {"level": entry.level, "confidence": entry.confidence}
            for name, entry in profile.skills.items()
        },
        "goals": [{"goal": g.goal, "importance": g.importance} for g in profile.goals],
    }


@app.get("/subjects")
async def get_subjects_endpoint():
    """Return the list of available learning subjects discovered from data/."""
    from src.rag.course_catalog import get_available_subjects_from_data

    return {"subjects": get_available_subjects_from_data()}


# ═══════════════════════════════════════════════════════════════════════════
# Analytics Endpoints — Growth, Cognitive Graph, Explainability, Dashboard
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/analytics/dashboard/{user_id}")
async def analytics_dashboard(user_id: str, subject: str = "", days: int = 30):
    """Return aggregated analytics dashboard data."""
    from src.analytics.memory_dashboard import get_dashboard_data
    from src.profile import get_profile_manager

    manager = get_profile_manager()
    profile = await manager.get_profile(user_id)
    data = await get_dashboard_data(
        user_id=user_id,
        profile=profile,
        subject=subject,
        days=days,
    )
    return data.model_dump()


@app.get("/analytics/growth/{user_id}")
async def analytics_growth(user_id: str, subject: str = "", days: int = 30):
    """Return skill growth time-series data."""
    from src.analytics.growth_analyzer import analyze_growth

    data = await analyze_growth(user_id=user_id, subject=subject, days=days)
    return data.model_dump()


@app.get("/analytics/cognitive-graph/{user_id}")
async def analytics_cognitive_graph(user_id: str, subject: str = ""):
    """Return cognitive model graph (nodes + edges)."""
    from src.analytics.cognitive_graph import build_cognitive_graph
    from src.profile import get_profile_manager

    manager = get_profile_manager()
    profile = await manager.get_profile(user_id)
    data = await build_cognitive_graph(
        user_id=user_id,
        profile=profile,
        subject=subject,
    )
    return data.model_dump()


@app.get("/analytics/decisions/{user_id}")
async def analytics_decisions(user_id: str, limit: int = 20, node: str = ""):
    """Return recent agent decision traces."""
    from src.analytics.explainability_engine import get_decision_traces

    node_name = node if node else None
    data = await get_decision_traces(
        user_id=user_id,
        limit=limit,
        node_name=node_name,
    )
    return data.model_dump()


if __name__ == "__main__":
    import uvicorn
    from src.config.server_runtime_config import (
        load_server_reload_config,
        resolve_uvicorn_reload_options,
    )

    reload_config = load_server_reload_config()
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        **resolve_uvicorn_reload_options(
            reload_config,
            workspace_root=Path(__file__).resolve().parent,
        ),
    )
