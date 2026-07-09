"""A3 Study Agent - AI-powered university learning resource generation system."""

from __future__ import annotations

# ruff: noqa: E402
# The application loads .env before importing project modules that read settings at import time.

import json
import logging
import os
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator
from urllib.parse import unquote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from src.context_engineering.schema import sanitize_error_message
from src.context_engineering.input_manifest import (
    background_context_status_payload,
    build_background_context_window,
    build_thread_context_ledger_update,
    llm_input_manifest_trace_payload,
    merge_llm_input_manifest_history,
)
from src.context_engineering.workspace import workspace_status_payload

load_dotenv(Path(__file__).parent / ".env")

from src.database.checkpointer import (
    checkpointer_enabled,
    checkpointer_type,
    get_db_uri,
    make_thread_config,
)
from src.config import get_setting
from src.graph.exercises import _render_exercise_markdown
from src.graph.builder import get_compiled_graph
from src.graph.state import (
    CONTEXT_CLEAR,
    DICT_CLEAR,
    GENERATED_ARTIFACTS_CLEAR,
    LLM_INPUT_MANIFESTS_CLEAR,
    MEMORY_CLEAR,
    TASK_WORKSPACE_CLEAR,
    WORKSPACE_EVENTS_CLEAR,
    initial_request_reset_transient_state,
)
from src.graph.resource_final import (
    completed_without_resource_payload,
    normalize_resource_final_payload,
)
from src.profile import get_profile_manager
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
    OnboardRequest,
    ResumeRequest,
    StopRequest,
    ThreadStatusResponse,
)
from src.observability.a3_trace import (
    emit_a3_trace,
    reset_trace_event_sink,
    set_trace_event_sink,
)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage async resources: tracing, PostgreSQL checkpointer, graph."""
    setup_tracing()

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

        app.state.checkpointer_enabled = bool(checkpointer)
        app.state.checkpointer_type = ckp_type
        graph = get_compiled_graph(checkpointer=checkpointer)
        setattr(graph, "_a3_checkpointer_enabled", bool(checkpointer))
        setattr(graph, "_a3_checkpointer_type", ckp_type)
        app.state.graph = graph
        yield

    shutdown_tracing()


app = FastAPI(title="A3 Study Agent API", lifespan=lifespan)

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


ALLOWED_NODES = {"generate_answer", "emotional_response"}

# Non-streaming nodes whose final AIMessage content is emitted as a "text" SSE event.
TEXT_EMIT_NODES = {
    "handle_unknown",
    "evidence_summary_output",
    "resource_bundle_output",
    "adaptive_practice_responder",
    "recommendation_provider",
}

# All graph nodes whose lifecycle (start/end) we broadcast to the frontend.
GRAPH_NODES = {
    "supervisor",
    "episodic_memory_retriever",
    "episodic_memory_writer",
    "memory_use_decider",
    "academic_router",
    "search_query_rewriter",
    "rag_retrieve",
    "web_search",
    "evidence_judge",
    "evidence_summary_output",
    "generate_answer",
    "evaluate_hallucination",
    "rewrite_query",
    "resource_orchestrator",
    "resource_worker",
    "resource_bundle_output",
    "curriculum_planner",
    "assessment_result_handler",
    "adaptive_practice_responder",
    "recommendation_provider",
    "emotional_response",
    "handle_unknown",
}


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
    last_resource_payload = (
        values.get("last_resource_final_payload")
        if isinstance(values.get("last_resource_final_payload"), dict)
        else {}
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
        "background_context_window": background_window,
        **background_status,
        "llm_input_manifest_count": len(manifest_history),
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


def _dict_keys(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(sanitize_error_message(key, max_chars=120) for key in value)


def _artifact_count(by_type: object, generated: object) -> int:
    total = len(by_type) if isinstance(by_type, dict) else 0
    total += len(generated) if isinstance(generated, list) else 0
    return total


def _new_request_status_values(snapshot_values: dict, initial_run_values: dict) -> dict:
    """Combine current run control with persistent thread context for status UI."""
    values = dict(snapshot_values or {})
    previous_history = values.get("context_usage_history")
    previous_manifests = values.get("llm_input_manifests")
    previous_manifest = values.get("llm_input_manifest")
    previous_ledger = values.get("thread_context_ledger")
    previous_background = values.get("background_context_window")
    previous_resource_payload = values.get("last_resource_final_payload")
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
            last_llm_input_manifest=_last_llm_input_manifest(values),
            llm_input_manifest_count=len(values.get("llm_input_manifests") or [])
            if isinstance(values.get("llm_input_manifests"), list)
            else 0,
            background_context_window=values.get("background_context_window")
            if isinstance(values.get("background_context_window"), dict)
            else {},
            last_resource_final_payload=values.get("last_resource_final_payload")
            if isinstance(values.get("last_resource_final_payload"), dict)
            else {},
            request_context_window=request_context_window,
            thread_context_window=thread_context_window,
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
        last_llm_input_manifest=_last_llm_input_manifest(values),
        llm_input_manifest_count=len(values.get("llm_input_manifests") or [])
        if isinstance(values.get("llm_input_manifests"), list)
        else 0,
        background_context_window=values.get("background_context_window")
        if isinstance(values.get("background_context_window"), dict)
        else {},
        last_resource_final_payload=values.get("last_resource_final_payload")
        if isinstance(values.get("last_resource_final_payload"), dict)
        else {},
        request_context_window=request_context_window,
        thread_context_window=thread_context_window,
        profile_completion_request=profile_completion_request,
        missing_run_control_fields=[],
    )


def _thread_status_from_active_run(
    thread_id: str, active_run: dict
) -> ThreadStatusResponse:
    request_context_window = active_run.get("request_context_window")
    thread_context_window = active_run.get("thread_context_window")
    profile_completion_request = active_run.get("profile_completion_request")
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
        last_llm_input_manifest=active_run.get("llm_input_manifest")
        if isinstance(active_run.get("llm_input_manifest"), dict)
        else {},
        llm_input_manifest_count=len(active_run.get("llm_input_manifests") or [])
        if isinstance(active_run.get("llm_input_manifests"), list)
        else 0,
        background_context_window=active_run.get("background_context_window")
        if isinstance(active_run.get("background_context_window"), dict)
        else {},
        last_resource_final_payload=active_run.get("last_resource_final_payload")
        if isinstance(active_run.get("last_resource_final_payload"), dict)
        else {},
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
            "workspace_present": False,
            "workspace_active_subject": "",
            "workspace_evidence_summary_count": 0,
            "workspace_gap_count": 0,
            "workspace_artifact_count": 0,
            "workspace_updated_at": "",
        },
        profile_completion_request=profile_completion_request
        if isinstance(profile_completion_request, dict)
        else {},
        missing_run_control_fields=[],
    )


def _valid_state_update_node(values: dict, state: dict | None = None) -> str:
    candidates = (
        values.get("current_node"),
        (state or {}).get("current_node"),
        (state or {}).get("last_completed_node"),
        "supervisor",
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text in GRAPH_NODES:
            return text
    return "supervisor"


async def safe_update_thread_state(
    graph,
    config: dict,
    values: dict,
    *,
    state: dict | None = None,
    as_node: str | None = None,
) -> None:
    """Update checkpoint state using a real graph node as LangGraph writer."""
    node = str(as_node or "").strip()
    if node not in GRAPH_NODES:
        node = _valid_state_update_node(values, state)
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
        if thread_id and get_active_run(thread_id) is not None:
            update_active_run(thread_id, values)
        emit_a3_trace(
            logger,
            "run_state_update_failed",
            {
                "keys": sorted(values.keys()),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
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


def _last_ai_message_content(final_state: dict) -> str:
    for msg in reversed(final_state.get("messages") or []):
        content = getattr(msg, "content", "")
        if content:
            return str(content)
    return ""


def _legacy_resource_final_payload(final_state: dict) -> dict | None:
    has_generated_resource_artifact = any(
        bool(final_state.get(key))
        for key in (
            "mindmap_artifact",
            "mindmap_tree",
            "exercise_artifact",
            "exercise_items",
            "review_doc_artifact",
            "review_doc_artifacts",
            "code_practice_artifact",
            "video_script_artifact",
            "video_animation_artifact",
            "study_plan_artifact",
            "study_plan_document_artifact",
            "resource_bundle_artifact",
        )
    )
    if (
        final_state.get("evidence_controlled_stop") is True
        or final_state.get("final_response_type") == "evidence_summary"
    ) and not has_generated_resource_artifact:
        answer = _last_ai_message_content(final_state) or str(
            final_state.get("plan") or ""
        )
        return {
            "type": "resource_final",
            "resource_type": "evidence_summary",
            "controlled_stop": True,
            "controlled_stop_reason": final_state.get(
                "evidence_controlled_stop_reason", ""
            ),
            "answer": answer,
        }

    bundle_artifact = final_state.get("resource_bundle_artifact") or {}
    requested_resource_types = list(final_state.get("requested_resource_types") or [])
    bundle_resources = list(bundle_artifact.get("resources") or [])
    bundle_errors = list(bundle_artifact.get("errors") or [])
    is_resource_bundle_payload = bool(bundle_artifact) and (
        len(requested_resource_types) > 1
        or len(bundle_resources) > 1
        or bundle_artifact.get("status") in {"partial_success", "failed"}
    )
    mindmap_artifact = final_state.get("mindmap_artifact") or {}
    mindmap_tree = final_state.get("mindmap_tree") or {}
    exercise_items = final_state.get("exercise_items") or []
    exercise_artifact = final_state.get("exercise_artifact") or {}
    review_doc_artifact = final_state.get("review_doc_artifact") or {}
    review_doc_artifacts = final_state.get("review_doc_artifacts") or []
    code_practice_artifact = final_state.get("code_practice_artifact") or {}
    video_script_artifact = final_state.get("video_script_artifact") or {}
    video_animation_artifact = final_state.get("video_animation_artifact") or {}
    study_plan_artifact = final_state.get("study_plan_artifact") or {}
    study_plan_document = final_state.get("study_plan_document_artifact") or {}

    if is_resource_bundle_payload:
        answer = _last_ai_message_content(final_state) or str(
            bundle_artifact.get("message") or ""
        )
        payload: dict = {
            "type": "resource_final",
            "resource_type": "bundle",
            "answer": answer,
            "resource_generation_status": final_state.get(
                "resource_generation_status", ""
            ),
            "resource_bundle": bundle_artifact,
            "resources": bundle_resources,
            "errors": bundle_errors,
        }

        if mindmap_artifact or mindmap_tree:
            payload["mindmap"] = {
                "title": mindmap_artifact.get("title", "Knowledge Mindmap"),
                "tree": (mindmap_artifact.get("tree") or mindmap_tree or {}),
                "xmind_url": mindmap_artifact.get("xmind_url", ""),
            }

        if exercise_items or exercise_artifact:
            if not answer and exercise_items:
                title = str(exercise_artifact.get("title") or "Leveled exercises")
                answer = _render_exercise_markdown(
                    title,
                    exercise_items,
                    review_reason=str(
                        exercise_artifact.get("review_reason")
                        or final_state.get("exercise_review_reason")
                        or ""
                    ),
                    quality_warning=bool(exercise_artifact.get("quality_warning")),
                )
                payload["answer"] = answer
            payload["exercise_items"] = exercise_items
            payload["exercise_artifact"] = exercise_artifact

        if review_doc_artifact:
            payload["review_doc"] = {
                "subject": review_doc_artifact.get("subject", ""),
                "title": review_doc_artifact.get("title", "Markdown Review Document"),
                "filename": review_doc_artifact.get("filename", ""),
                "docx_filename": review_doc_artifact.get("docx_filename", ""),
                "markdown_url": review_doc_artifact.get("markdown_url", ""),
                "docx_url": review_doc_artifact.get("docx_url", ""),
                "markdown": review_doc_artifact.get("markdown", ""),
            }
        if review_doc_artifacts:
            payload["review_doc_artifacts"] = [
                {
                    "subject": artifact.get("subject", ""),
                    "title": artifact.get("title", "Markdown复习文档"),
                    "filename": artifact.get("filename", ""),
                    "docx_filename": artifact.get("docx_filename", ""),
                    "markdown_url": artifact.get("markdown_url", ""),
                    "docx_url": artifact.get("docx_url", ""),
                    "markdown": artifact.get("markdown", ""),
                }
                for artifact in review_doc_artifacts
            ]

        if code_practice_artifact:
            payload["code_practice_artifact"] = code_practice_artifact
        if video_script_artifact:
            payload["video_script_artifact"] = video_script_artifact
        if video_animation_artifact:
            payload["video_animation_artifact"] = video_animation_artifact

        if study_plan_artifact or study_plan_document:
            payload["study_plan"] = {
                "title": study_plan_artifact.get("title")
                or study_plan_document.get("title", "Personalized Study Plan"),
                "filename": study_plan_document.get("filename", ""),
                "docx_filename": study_plan_document.get("docx_filename", ""),
                "markdown_url": study_plan_document.get("markdown_url", ""),
                "docx_url": study_plan_document.get("docx_url", ""),
                "markdown": (
                    final_state.get("study_plan_markdown", "")
                    or study_plan_document.get("markdown", "")
                ),
            }

        return payload

    resource_type = str(final_state.get("requested_resource_type") or "")

    if resource_type not in {
        "mindmap",
        "quiz",
        "review_doc",
        "code_practice",
        "video_script",
        "video_animation",
        "study_plan",
        "bundle",
    }:
        if mindmap_artifact or mindmap_tree:
            resource_type = "mindmap"
        elif exercise_items or exercise_artifact:
            resource_type = "quiz"
        elif review_doc_artifact or review_doc_artifacts:
            resource_type = "review_doc"
        elif code_practice_artifact:
            resource_type = "code_practice"
        elif video_script_artifact:
            resource_type = "video_script"
        elif video_animation_artifact:
            resource_type = "video_animation"
        elif study_plan_artifact or study_plan_document:
            resource_type = "study_plan"
        else:
            return None

    answer = _last_ai_message_content(final_state)
    payload = {
        "type": "resource_final",
        "resource_type": resource_type,
        "answer": answer,
    }

    include_mindmap = resource_type == "mindmap" and (mindmap_artifact or mindmap_tree)
    include_quiz = resource_type == "quiz" and (exercise_items or exercise_artifact)
    include_review_doc = resource_type == "review_doc" and review_doc_artifact
    include_review_doc_artifacts = (
        resource_type == "review_doc" and review_doc_artifacts
    )
    include_code_practice = resource_type == "code_practice" and code_practice_artifact
    include_video_script = resource_type == "video_script" and video_script_artifact
    include_video_animation = (
        resource_type == "video_animation" and video_animation_artifact
    )

    if include_mindmap:
        payload["mindmap"] = {
            "title": mindmap_artifact.get("title", "Knowledge Mindmap"),
            "tree": (mindmap_artifact.get("tree") or mindmap_tree or {}),
            "xmind_url": mindmap_artifact.get("xmind_url", ""),
        }
        payload["mindmap_artifact"] = mindmap_artifact
        payload["mindmap_tree"] = mindmap_artifact.get("tree") or mindmap_tree or {}

    if include_quiz:
        if (
            not answer or (resource_type == "quiz" and len(answer.strip()) < 40)
        ) and exercise_items:
            title = str(exercise_artifact.get("title") or "Leveled exercises")
            answer = _render_exercise_markdown(
                title,
                exercise_items,
                review_reason=str(
                    exercise_artifact.get("review_reason")
                    or final_state.get("exercise_review_reason")
                    or ""
                ),
                quality_warning=bool(exercise_artifact.get("quality_warning")),
            )
            payload["answer"] = answer
        payload["exercise_items"] = exercise_items
        payload["exercise_artifact"] = exercise_artifact

    if include_review_doc:
        payload["review_doc"] = {
            "subject": review_doc_artifact.get("subject", ""),
            "title": review_doc_artifact.get("title", "Markdown Review Document"),
            "filename": review_doc_artifact.get("filename", ""),
            "docx_filename": review_doc_artifact.get("docx_filename", ""),
            "markdown_url": review_doc_artifact.get("markdown_url", ""),
            "docx_url": review_doc_artifact.get("docx_url", ""),
            "markdown": review_doc_artifact.get("markdown", ""),
        }
        payload["review_doc_artifact"] = review_doc_artifact
    if include_review_doc_artifacts:
        payload["review_doc_artifacts"] = [
            {
                "subject": artifact.get("subject", ""),
                "title": artifact.get("title", "Markdown复习文档"),
                "filename": artifact.get("filename", ""),
                "docx_filename": artifact.get("docx_filename", ""),
                "markdown_url": artifact.get("markdown_url", ""),
                "docx_url": artifact.get("docx_url", ""),
                "markdown": artifact.get("markdown", ""),
            }
            for artifact in review_doc_artifacts
        ]

    if include_code_practice:
        payload["code_practice_artifact"] = code_practice_artifact
    if include_video_script:
        payload["video_script_artifact"] = video_script_artifact
    if include_video_animation:
        payload["video_animation_artifact"] = video_animation_artifact

    if resource_type == "study_plan" and (study_plan_artifact or study_plan_document):
        payload["study_plan"] = {
            "title": study_plan_artifact.get("title")
            or study_plan_document.get("title", "Personalized Study Plan"),
            "filename": study_plan_document.get("filename", ""),
            "docx_filename": study_plan_document.get("docx_filename", ""),
            "markdown_url": study_plan_document.get("markdown_url", ""),
            "docx_url": study_plan_document.get("docx_url", ""),
            "markdown": (
                final_state.get("study_plan_markdown", "")
                or study_plan_document.get("markdown", "")
            ),
        }

    return payload


def _resource_final_payload(final_state: dict) -> dict | None:
    """Build a stable resource_final event while preserving legacy fields."""
    legacy_payload = _legacy_resource_final_payload(final_state)
    return normalize_resource_final_payload(legacy_payload, final_state)


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
        "evidence_summary_memory",
        "evidence_gap_memory",
        "episodic_memory_results",
        "semantic_memory_results",
        "task_workspace",
        "workspace_events",
        "resource_artifacts_by_type",
        "last_generated_artifacts",
        "last_resource_final_payload",
        "llm_input_manifest",
        "llm_input_manifests",
        "thread_context_ledger",
        "background_context_window",
        "context_continuity",
    ]
    values = {
        "conversation_summary": "",
        "evidence_summary_memory": MEMORY_CLEAR,
        "evidence_gap_memory": MEMORY_CLEAR,
        "episodic_memory_results": [],
        "semantic_memory_results": [],
        "task_workspace": TASK_WORKSPACE_CLEAR,
        "workspace_events": WORKSPACE_EVENTS_CLEAR,
        "resource_artifacts_by_type": DICT_CLEAR,
        "last_generated_artifacts": GENERATED_ARTIFACTS_CLEAR,
        "last_resource_final_payload": DICT_CLEAR,
        "llm_input_manifest": {},
        "llm_input_manifests": LLM_INPUT_MANIFESTS_CLEAR,
        "thread_context_ledger": DICT_CLEAR,
        "background_context_window": {},
        "context_continuity": {},
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


async def _stream_graph_events(
    graph,
    input_data,
    config: dict,
    thread_id: str,
    preserve_context_history: bool = False,
) -> AsyncGenerator[str, None]:
    """Shared SSE event streaming logic for /stream and /resume.

    Processes astream_events and yields SSE payloads for node lifecycle,
    token streaming, usage, and interrupt events.
    """
    node_start_times: dict[str, float] = {}
    active_nodes: list[str] = []
    trace_events: list[dict] = []
    trace_sink_token = set_trace_event_sink(trace_events)
    context_usage_history: list[dict] = []
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

    async def _drain_trace_events() -> list[str]:
        nonlocal llm_input_manifests, manifest_state_context
        drained: list[str] = []
        while trace_events:
            event = trace_events.pop(0)
            stage = event.get("stage")
            if isinstance(stage, str) and stage.startswith("context_"):
                request_context_events.append(_safe_context_event_summary(event))
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
                continue
            if stage == "context_policy_resolved":
                payload = _context_policy_resolved_payload(event)
                node = str(payload.get("node") or "")
                if node:
                    last_context_policy_by_node[node] = payload
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
                continue
            if stage == "context_provider_supply_plan":
                payload = _context_provider_supply_plan_payload(event)
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
                continue
            if stage == "context_provider_supply":
                payload = _context_provider_supply_payload(event)
                node = str(payload.get("node") or "")
                if node:
                    last_provider_supply_by_node[node] = payload
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
                continue
            if stage == "context_source_filter":
                payload = _context_source_filter_payload(event)
                node = str(payload.get("node") or "")
                if node:
                    last_drop_reasons_by_node[node] = payload.get("drop_reasons", {})
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(
                    f"data: {json.dumps(sse_payload, ensure_ascii=False)}\n\n"
                )
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
                continue
            if stage == "context_applied":
                payload = {
                    "type": "context_applied",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
                    "trace_call_id": event.get("trace_call_id", ""),
                    "trace_seq": event.get("trace_seq", 0),
                    "applied": bool(event.get("applied", False)),
                    "fallback_used": bool(event.get("fallback_used", False)),
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                    "fallback_used": bool(event.get("fallback_used", False)),
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                drained.append(
                    f"data: {json.dumps(context_error_payload, ensure_ascii=False)}\n\n"
                )
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
                drained.append(
                    f"data: {json.dumps(window_payload, ensure_ascii=False)}\n\n"
                )
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
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
                    "fallback_to_rule_based": bool(
                        event.get("fallback_to_rule_based", False)
                    ),
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
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
                continue
        return drained

    try:
        async for event in graph.astream_events(
            input_data, config=config, version="v2"
        ):
            for trace_payload in await _drain_trace_events():
                yield trace_payload
            event_type = event["event"]

            # Node lifecycle events
            if event_type in ("on_chain_start", "on_chain_end"):
                node_name = event.get("name")
                meta_node = event.get("metadata", {}).get("langgraph_node")
                # Only emit for top-level graph nodes (name matches metadata),
                # not for internal sub-chains (RunnableSequence, etc.).
                if node_name and node_name == meta_node and node_name in GRAPH_NODES:
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
                        payload = json.dumps(
                            {
                                "type": "node_event",
                                "status": "start",
                                "node": node_name,
                            },
                            ensure_ascii=False,
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
                            and isinstance(output.get("resource_bundle_artifact"), dict)
                            and output.get("resource_bundle_artifact")
                        ):
                            terminal_resource_output = output

                        payload = json.dumps(
                            {
                                "type": "node_event",
                                "status": "end",
                                "node": node_name,
                                "duration_ms": duration_ms,
                                "error": error,
                            },
                            ensure_ascii=False,
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
                    yield f"data: {payload}\n\n"

                    # Emit "text" for non-streaming nodes (AC-02)
                    if event_type == "on_chain_end" and node_name in TEXT_EMIT_NODES:
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict):
                            for msg in output.get("messages", []):
                                if hasattr(msg, "content") and msg.content:
                                    text_payload = json.dumps(
                                        {
                                            "type": "text",
                                            "content": msg.content,
                                            "node": node_name,
                                        },
                                        ensure_ascii=False,
                                    )
                                    yield f"data: {text_payload}\n\n"

                    for trace_payload in await _drain_trace_events():
                        yield trace_payload

            # Token streaming
            elif event_type == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name in ALLOWED_NODES:
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        payload = json.dumps(
                            {"type": "token", "content": chunk.content},
                            ensure_ascii=False,
                        )
                        yield f"data: {payload}\n\n"

            # Token usage events
            elif event_type == "on_chat_model_end":
                node_name = event.get("metadata", {}).get("langgraph_node")
                output = event.get("data", {}).get("output")
                usage = getattr(output, "usage_metadata", None)
                if usage and node_name:
                    payload = json.dumps(
                        {
                            "type": "usage",
                            "node": node_name,
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                        ensure_ascii=False,
                    )
                    yield f"data: {payload}\n\n"
        for trace_payload in await _drain_trace_events():
            yield trace_payload
    except Exception as e:
        for trace_payload in await _drain_trace_events():
            yield trace_payload
        logger.exception("Unhandled error in graph streaming")
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
        run_control_registry.clear_stop_signal(thread_id)
        failed_node = active_nodes[-1] if active_nodes else None
        if failed_node:
            start_t = node_start_times.get(failed_node)
            duration_ms = (
                round((time.monotonic() - start_t) * 1000)
                if start_t is not None
                else None
            )
            node_payload = json.dumps(
                {
                    "type": "node_event",
                    "status": "end",
                    "node": failed_node,
                    "duration_ms": duration_ms,
                    "error": str(e),
                    "synthetic": True,
                },
                ensure_ascii=False,
            )
            yield f"data: {node_payload}\n\n"
        error_payload = json.dumps(
            {
                "type": "error",
                "message": str(e),
                "failed_node": failed_node,
                "active_nodes": active_nodes,
            },
            ensure_ascii=False,
        )
        yield f"data: {error_payload}\n\n"
        if failed_update_ok:
            finish_active_run(thread_id, {"run_status": RUN_STATUS_ERROR})
        return
    finally:
        reset_trace_event_sink(trace_sink_token)

    # Check for interrupt after stream completes
    try:
        state_snapshot = await graph.aget_state(config)
    except Exception:
        logger.exception("Failed to read graph state snapshot after stream")
        raise

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
        state=final_state,
        env_flag="LOG_A3_TRACE",
    )

    if state_snapshot.next:
        for task in state_snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_value = task.interrupts[0].value
                if (
                    isinstance(interrupt_value, dict)
                    and interrupt_value.get("type") == "user_stop"
                ):
                    stopped_at = utc_now_iso()
                    stopped_node = str(interrupt_value.get("node") or "")
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
                            "stop_reason": str(
                                interrupt_value.get("reason") or "user_stop"
                            ),
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
                    payload = json.dumps(
                        {
                            "type": "run_status",
                            "run_status": RUN_STATUS_STOPPED,
                            "thread_id": thread_id,
                            "resume_available": True,
                            "pending_interrupt_type": "user_stop",
                            "node": stopped_node,
                            "stopped_at": stopped_at,
                        },
                        ensure_ascii=False,
                    )
                    yield f"data: {payload}\n\n"
                    if stopped_update_ok:
                        finish_active_run(thread_id, {"run_status": RUN_STATUS_STOPPED})
                    return
                if (
                    isinstance(interrupt_value, dict)
                    and interrupt_value.get("type") == "memory_confirmation"
                ):
                    interrupt_update_ok = await _try_update_run_state(
                        graph,
                        config,
                        {
                            "resume_available": False,
                            "pending_interrupt_type": "memory_confirmation",
                        },
                        state=final_state,
                        persist_checkpoint=False,
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
                    interrupt_update_ok = await _try_update_run_state(
                        graph,
                        config,
                        {
                            "resume_available": True,
                            "pending_interrupt_type": "profile_completion_required",
                            "profile_completion_request": profile_request,
                        },
                        state=final_state,
                        persist_checkpoint=False,
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
                    interrupt_update_ok = await _try_update_run_state(
                        graph,
                        config,
                        {
                            "resume_available": False,
                            "pending_interrupt_type": "plan_review",
                        },
                        state=final_state,
                        persist_checkpoint=False,
                    )
                    payload_data = {
                        "type": "interrupt",
                        "interrupt_type": "plan_review",
                        "draft": interrupt_value,
                        "thread_id": thread_id,
                    }
                payload = json.dumps(payload_data, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                if interrupt_update_ok:
                    finish_active_run(thread_id, {"run_status": RUN_STATUS_STOPPED})
                return

    final_request_context_window = {
        "current_request_id": request_context_events[-1].get("request_id", "")
        if request_context_events
        else str(final_state.get("request_id") or ""),
        "current_node": "",
        "last_event_count": len(request_context_events),
    }
    resource_payload = _resource_final_payload(final_state)
    if terminal_resource_output:
        terminal_resource_payload = _resource_final_payload(terminal_resource_output)
        if terminal_resource_payload:
            resource_payload = terminal_resource_payload
    diagnostic_state = terminal_resource_output or final_state
    resource_diagnostic_payload = (
        completed_without_resource_payload(diagnostic_state) if not resource_payload else None
    )
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
    }
    if resource_payload:
        completed_values["last_resource_final_payload"] = resource_payload
    if context_usage_history:
        completed_values["context_usage"] = context_usage_history[-1]
        completed_values["context_usage_history"] = list(context_usage_history)
    if llm_input_manifests:
        completed_values["llm_input_manifest"] = llm_input_manifests[0]
        completed_values["llm_input_manifests"] = list(llm_input_manifests)
        background_window = manifest_state_context.get("background_context_window")
        if isinstance(background_window, dict):
            completed_values["background_context_window"] = background_window
            active_thread_window = (
                dict(completed_values.get("thread_context_window") or {})
                if isinstance(completed_values.get("thread_context_window"), dict)
                else {}
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

    completed_update_ok = await _try_update_run_state(
        graph,
        config,
        completed_values,
        state=final_state,
    )
    run_control_registry.clear_stop_signal(thread_id)
    if resource_payload:
        emit_a3_trace(
            logger,
            "sse_resource_final",
            {
                "sent": True,
                "resource_id": resource_payload.get("resource_id", ""),
                "payload_hash": resource_payload.get("payload_hash", ""),
                "resource_type": resource_payload.get("resource_type", ""),
                "answer_chars": len(str(resource_payload.get("answer") or "")),
                "has_mindmap": bool(resource_payload.get("mindmap")),
                "has_review_doc": bool(resource_payload.get("review_doc")),
                "has_exercise": bool(resource_payload.get("exercise_artifact")),
                "has_code_practice": bool(
                    resource_payload.get("code_practice_artifact")
                ),
                "has_video_script": bool(resource_payload.get("video_script_artifact")),
                "has_video_animation": bool(
                    resource_payload.get("video_animation_artifact")
                ),
                "video_animation_render_success": bool(
                    (resource_payload.get("video_animation_artifact") or {}).get(
                        "render_success"
                    )
                ),
                "review_doc_artifacts_count": len(
                    resource_payload.get("review_doc_artifacts") or []
                ),
                "has_review_doc_artifacts": bool(
                    resource_payload.get("review_doc_artifacts")
                ),
                "exercise_items_count": len(
                    resource_payload.get("exercise_items") or []
                ),
                "controlled_stop": bool(resource_payload.get("controlled_stop")),
            },
            state=final_state,
            env_flag="LOG_A3_TRACE",
        )
        yield f"data: {json.dumps(resource_payload, ensure_ascii=False)}\n\n"
    elif resource_diagnostic_payload:
        emit_a3_trace(
            logger,
            "sse_resource_final",
            {
                "sent": False,
                "status": "completed_without_resource",
                "resource_generation_status": resource_diagnostic_payload.get(
                    "resource_generation_status", ""
                ),
                "requested_resource_types": resource_diagnostic_payload.get(
                    "requested_resource_types", []
                ),
            },
            state=final_state,
            env_flag="LOG_A3_TRACE",
        )
        yield f"data: {json.dumps(resource_diagnostic_payload, ensure_ascii=False)}\n\n"

    completed_payload = {
        "type": "run_status",
        "run_status": RUN_STATUS_COMPLETED,
        "thread_id": thread_id,
    }
    yield f"data: {json.dumps(completed_payload, ensure_ascii=False)}\n\n"

    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
    if completed_update_ok:
        finish_active_run(thread_id, {"run_status": RUN_STATUS_COMPLETED})


async def generate_sse(
    query: str,
    graph,
    thread_id: str | None = None,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream LangGraph events as Server-Sent Events (SSE).

    Yields SSE payload types:

    * ``{"type": "thread_id", "thread_id": "..."}``
      - emitted once at stream start so frontend can use it for /resume.
    * ``{"type": "node_event", "status": "start"|"end", "node": "<name>"}``
      - emitted when a graph node begins or finishes execution.
    * ``{"type": "token", "content": "<text>"}``
      - emitted for each streamed token from an allowed LLM node.
    * ``{"type": "interrupt", "draft": "...", "thread_id": "..."}``
      - emitted when the graph pauses for human review (HIL).

    Args:
        query: The user-provided string to be processed by the graph.
        graph: The compiled LangGraph instance from app.state.
        thread_id: Optional session ID for multi-turn memory. Auto-generated if None.
        user_id: Optional user ID for profile context injection and recording.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
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
        payload = {
            "type": "error",
            "message": "thread_checkpoint_initialization_failed",
            "recoverable": False,
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return

    run_state_snapshot = await graph.aget_state(config)
    status_values = _new_request_status_values(
        _state_values(run_state_snapshot),
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
            "context_usage": {},
            "context_usage_history": status_values.get("context_usage_history")
            if isinstance(status_values.get("context_usage_history"), list)
            else [],
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
            "last_resource_final_payload": status_values.get(
                "last_resource_final_payload"
            )
            if isinstance(status_values.get("last_resource_final_payload"), dict)
            else {},
        },
    )

    thread_payload = {"type": "thread_id", "thread_id": thread_id}
    yield f"data: {json.dumps(thread_payload, ensure_ascii=False)}\n\n"
    running_payload = {
        "type": "run_status",
        "run_status": RUN_STATUS_RUNNING,
        "thread_id": thread_id,
    }
    yield f"data: {json.dumps(running_payload, ensure_ascii=False)}\n\n"

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
                {"thread_id": thread_id},
                memory_type=mem_type,
                content=content,
                importance=importance,
            )
        except Exception:
            logger.exception("Failed to record user input episodic memory")

    async for chunk in _stream_graph_events(
        graph, state_input, config, thread_id, preserve_context_history=True
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


async def generate_resume_sse(
    edited_plan: str,
    feedback: str | None,
    graph,
    thread_id: str,
    memory_use_choice: str | None = None,
    profile_completion: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Resume an interrupted graph and stream remaining events as SSE.

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
        payload = {
            "type": "run_status",
            "run_status": RUN_STATUS_STOPPED,
            "thread_id": thread_id,
            "resume_available": True,
            "pending_interrupt_type": "user_stop",
            "message": "use /threads/{thread_id}/continue for user_stop interrupts",
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
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

    resume_input = Command(resume=resume_value)
    resume_request_id = str(uuid.uuid4())
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
            "context_usage": status_values.get("context_usage")
            if isinstance(status_values.get("context_usage"), dict)
            else {},
            "context_usage_history": status_values.get("context_usage_history")
            if isinstance(status_values.get("context_usage_history"), list)
            else [],
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
            "last_resource_final_payload": status_values.get(
                "last_resource_final_payload"
            )
            if isinstance(status_values.get("last_resource_final_payload"), dict)
            else {},
        },
    )

    continuing_payload = {
        "type": "run_status",
        "run_status": RUN_STATUS_CONTINUING,
        "thread_id": thread_id,
    }
    yield f"data: {json.dumps(continuing_payload, ensure_ascii=False)}\n\n"

    async for chunk in _stream_graph_events(
        graph, resume_input, config, thread_id, preserve_context_history=True
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


async def generate_continue_sse(graph, thread_id: str) -> AsyncGenerator[str, None]:
    config = make_thread_config(thread_id)
    state_snapshot = await graph.aget_state(config)
    if not _has_checkpoint_state(state_snapshot):
        payload = {
            "type": "run_status",
            "run_status": RUN_STATUS_NOT_RESUMABLE,
            "thread_id": thread_id,
            "resume_available": False,
            "message": "checkpoint_not_found",
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return

    pending_type = _pending_interrupt_type(state_snapshot)
    if pending_type in {"plan_review", "memory_confirmation"}:
        payload = {
            "type": "run_status",
            "run_status": RUN_STATUS_NOT_RESUMABLE,
            "thread_id": thread_id,
            "resume_available": False,
            "pending_interrupt_type": pending_type,
            "message": "pending HIL interrupt must be resumed with /resume",
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return
    if pending_type != "user_stop":
        payload = {
            "type": "run_status",
            "run_status": RUN_STATUS_NOT_RESUMABLE,
            "thread_id": thread_id,
            "resume_available": False,
            "pending_interrupt_type": pending_type,
            "message": "no pending user_stop interrupt",
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
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
            "context_usage": status_values.get("context_usage")
            if isinstance(status_values.get("context_usage"), dict)
            else {},
            "context_usage_history": status_values.get("context_usage_history")
            if isinstance(status_values.get("context_usage_history"), list)
            else [],
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
            "last_resource_final_payload": status_values.get(
                "last_resource_final_payload"
            )
            if isinstance(status_values.get("last_resource_final_payload"), dict)
            else {},
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
    continuing_payload = {
        "type": "run_status",
        "run_status": RUN_STATUS_CONTINUING,
        "thread_id": thread_id,
    }
    yield f"data: {json.dumps(continuing_payload, ensure_ascii=False)}\n\n"

    resume_input = Command(resume={"type": "user_stop", "action": "continue"})
    _emit_graph_config_trace(
        graph,
        config,
        {
            "request_id": str(uuid.uuid4()),
            "session_id": thread_id,
            "thread_id": thread_id,
        },
    )
    async for chunk in _stream_graph_events(
        graph, resume_input, config, thread_id, preserve_context_history=True
    ):
        yield chunk


@app.post("/stream")
async def stream_endpoint(chat: ChatRequest, request: Request):
    return StreamingResponse(
        generate_sse(
            chat.query,
            request.app.state.graph,
            thread_id=chat.thread_id,
            user_id=chat.user_id,
        ),
        media_type="text/event-stream",
    )


@app.post("/resume")
async def resume_endpoint(req: ResumeRequest, request: Request):
    profile_completion = (
        req.profile_completion.model_dump()
        if req.profile_completion is not None
        else None
    )
    return StreamingResponse(
        generate_resume_sse(
            req.edited_plan,
            req.feedback,
            request.app.state.graph,
            req.thread_id,
            memory_use_choice=req.memory_use_choice,
            profile_completion=profile_completion,
        ),
        media_type="text/event-stream",
    )


@app.post("/threads/{thread_id}/stop")
async def stop_thread_endpoint(thread_id: str, req: StopRequest, request: Request):
    return await request_thread_stop(request.app.state.graph, thread_id, req.reason)


@app.get("/threads/{thread_id}/status", response_model=ThreadStatusResponse)
async def thread_status_endpoint(thread_id: str, request: Request):
    return await get_thread_status_payload(request.app.state.graph, thread_id)


@app.post("/threads/{thread_id}/continue")
async def continue_thread_endpoint(thread_id: str, request: Request):
    return StreamingResponse(
        generate_continue_sse(request.app.state.graph, thread_id),
        media_type="text/event-stream",
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


@app.post("/onboard")
async def onboard_endpoint(req: OnboardRequest):
    """Create an initial user profile from the onboarding wizard.

    All values are explicit self-reports -> stored with confidence=0.9.
    """
    from src.profile.schema import (
        AgentObservation,
        Goal,
        LearningStyle,
        SkillEntry,
        UserProfile,
    )

    manager = get_profile_manager()
    now = datetime.now(timezone.utc).isoformat()

    # Build skills from self-assessed levels
    skills: dict[str, SkillEntry] = {}
    for subject in req.subjects:
        level = req.skill_levels.get(subject, 0.25)
        skills[subject] = SkillEntry(
            level=level,
            confidence=0.9,
            last_observed=now,
            evidence_count=1,
        )

    # Build learning style from self-report
    learning_style = LearningStyle()
    for dim, val in req.learning_style.items():
        if hasattr(learning_style, dim):
            setattr(learning_style, dim, val)

    # Build goals
    goals = [
        Goal(goal=g.strip(), importance=0.9, progress=0.0, created_at=now)
        for g in req.goals
        if g.strip()
    ]

    # Record observations about the onboarding
    obs_list: list[AgentObservation] = []
    if req.grade:
        obs_list.append(
            AgentObservation(
                content=f"用户自述年级: {req.grade}",
                category="general",
                importance=0.8,
                created_at=now,
            )
        )
    if req.subjects:
        obs_list.append(
            AgentObservation(
                content=f"用户首次选择 {len(req.subjects)} 个学习方向: {', '.join(req.subjects)}",
                category="general",
                importance=0.8,
                created_at=now,
            )
        )

    profile = UserProfile(
        user_id=req.user_id,
        skills=skills,
        learning_style=learning_style,
        goals=goals,
        dislikes=req.dislikes or [],
        agent_observations=obs_list,
        extra={
            "nickname": req.nickname,
            "grade": req.grade,
            "onboarding_completed": True,
        },
        created_at=now,
        updated_at=now,
    )

    await manager.store.save(profile)
    logger.info(
        "Onboarding 用户画像已创建 user=%s nickname=%s subjects=%d goals=%d",
        req.user_id,
        req.nickname,
        len(skills),
        len(goals),
    )

    return {
        "user_id": req.user_id,
        "summary": profile.to_summary(),
        "skills_count": len(skills),
        "goals_count": len(goals),
    }


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

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
