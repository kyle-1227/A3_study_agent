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
    MEMORY_CLEAR,
    initial_request_reset_transient_state,
)
from src.profile import get_profile_manager
from src.run_control import (
    RUN_CONTROL_FIELDS,
    RUN_CONTROL_SCHEMA_VERSION,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_CONTINUING,
    RUN_STATUS_ERROR,
    RUN_STATUS_NOT_RESUMABLE,
    RUN_STATUS_RUNNING,
    RUN_STATUS_STOPPED,
    RUN_STATUS_STOPPING,
    RUN_STATUS_UNKNOWN,
    run_control_registry,
    trim_context_usage_history,
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
        safe_item = {
            key: item[key]
            for key in PACKING_PREVIEW_FIELDS
            if key in item
        }
        if "title" in safe_item:
            safe_item["title"] = sanitize_error_message(
                safe_item["title"],
                max_chars=120,
            )
        safe_items.append(safe_item)
    return safe_items


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

        if enabled and ckp_type == "postgres" and db_uri:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                checkpointer = await stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(db_uri)
                )
                await checkpointer.setup()
                logger.info("PostgreSQL checkpointer initialized")
            except Exception:
                logger.exception(
                    "Failed to initialize PostgreSQL checkpointer, falling back to MemorySaver"
                )
                from langgraph.checkpoint.memory import MemorySaver

                checkpointer = MemorySaver()
                ckp_type = "memory"
        elif enabled:
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()
            ckp_type = "memory"
            if db_uri and checkpointer_type() == "postgres":
                logger.warning(
                    "DB_URI is set but PostgreSQL checkpointer was unavailable; using MemorySaver"
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
    "mindmap_output",
    "exercise_output",
    "review_doc_output",
    "code_practice_output",
    "video_script_output",
    "video_animation_output",
    "study_plan_output",
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
    "mindmap_planner",
    "mindmap_agent",
    "mindmap_reviewer",
    "mindmap_rewrite",
    "mindmap_output",
    "exercise_planner",
    "exercise_agent",
    "exercise_reviewer",
    "exercise_rewrite",
    "exercise_output",
    "review_doc_planner",
    "review_doc_agent",
    "review_doc_reviewer",
    "review_doc_rewrite",
    "review_doc_output",
    "code_practice_planner",
    "code_practice_agent",
    "code_practice_reviewer",
    "code_practice_rewrite",
    "code_practice_output",
    "video_script_planner",
    "video_script_agent",
    "video_script_reviewer",
    "video_script_rewrite",
    "video_script_output",
    "video_animation_planner",
    "video_animation_agent",
    "video_animation_reviewer",
    "video_animation_rewrite",
    "video_animation_output",
    "curriculum_planner",
    "assessment_result_handler",
    "adaptive_practice_responder",
    "recommendation_provider",
    "study_plan_emotional_intel",
    "study_plan_planner",
    "study_plan_agent",
    "study_plan_reviewer_academic",
    "study_plan_reviewer_emotional",
    "study_plan_consensus",
    "study_plan_rewrite",
    "study_plan_output",
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


def _has_checkpoint_state(state_snapshot) -> bool:
    values = _state_values(state_snapshot)
    if values:
        return True
    if getattr(state_snapshot, "next", None):
        return True
    return bool(_pending_interrupt_values(state_snapshot))


def _missing_run_control_fields(values: dict) -> list[str]:
    return [field for field in RUN_CONTROL_FIELDS if field not in values]


def _thread_status_from_snapshot(
    thread_id: str, state_snapshot
) -> ThreadStatusResponse:
    values = _state_values(state_snapshot)
    pending_type = _pending_interrupt_type(state_snapshot)
    missing_fields = _missing_run_control_fields(values)
    if missing_fields:
        return ThreadStatusResponse(
            thread_id=thread_id,
            schema_version="legacy",
            run_status=RUN_STATUS_UNKNOWN,
            resume_available=False,
            pending_interrupt_type=pending_type,
            current_node=str(values.get("current_node") or ""),
            last_completed_node=str(values.get("last_completed_node") or ""),
            context_usage=values.get("context_usage")
            if isinstance(values.get("context_usage"), dict)
            else {},
            context_usage_history=values.get("context_usage_history")
            if isinstance(values.get("context_usage_history"), list)
            else [],
            missing_run_control_fields=missing_fields,
            message="legacy checkpoint does not include run-control fields",
        )

    return ThreadStatusResponse(
        thread_id=thread_id,
        schema_version=RUN_CONTROL_SCHEMA_VERSION,
        run_status=str(values.get("run_status") or RUN_STATUS_UNKNOWN),
        resume_available=pending_type == "user_stop",
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
        missing_run_control_fields=[],
    )


async def _update_run_state(graph, config: dict, values: dict) -> None:
    await graph.aupdate_state(config, values)


async def _try_update_run_state(
    graph, config: dict, values: dict, *, state: dict | None = None
) -> bool:
    try:
        await _update_run_state(graph, config, values)
        return True
    except Exception as exc:
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


def _last_ai_message_content(final_state: dict) -> str:
    for msg in reversed(final_state.get("messages") or []):
        content = getattr(msg, "content", "")
        if content:
            return str(content)
    return ""


def _resource_final_payload(final_state: dict) -> dict | None:
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
    payload: dict = {
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
    ]
    values = {
        "conversation_summary": "",
        "evidence_summary_memory": MEMORY_CLEAR,
        "evidence_gap_memory": MEMORY_CLEAR,
        "episodic_memory_results": [],
        "semantic_memory_results": [],
    }
    await graph.aupdate_state(config, values)

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
    if preserve_context_history:
        try:
            existing_snapshot = await graph.aget_state(config)
            existing_history = _state_values(existing_snapshot).get(
                "context_usage_history"
            )
            if isinstance(existing_history, list):
                context_usage_history = trim_context_usage_history(existing_history)
        except Exception as exc:
            emit_a3_trace(
                logger,
                "run_state_read_failed",
                {
                    "operation": "load_context_usage_history",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                state={"thread_id": thread_id, "session_id": thread_id},
                env_flag="LOG_A3_TRACE",
            )

    async def _drain_trace_events() -> list[str]:
        drained: list[str] = []
        while trace_events:
            event = trace_events.pop(0)
            stage = event.get("stage")
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
            if stage == "context_usage":
                payload = {
                    "type": "context_usage",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
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
                await _try_update_run_state(
                    graph,
                    config,
                    {
                        "context_usage": payload,
                        "context_usage_history": list(context_usage_history),
                    },
                    state={"thread_id": thread_id, "session_id": thread_id},
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
                    "provider_count": event.get("provider_count", 0),
                    "item_count": event.get("item_count", 0),
                    "source_counts": event.get("source_counts")
                    if isinstance(event.get("source_counts"), dict)
                    else {},
                    "total_estimated_tokens": event.get("total_estimated_tokens", 0),
                    "top_items": _safe_context_top_items(event.get("top_items")),
                }
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
                continue
            if stage == "context_provider_error":
                payload = {
                    "type": "context_provider_error",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
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
                    "candidate_count": event.get("candidate_count", 0),
                    "source_counts": event.get("source_counts")
                    if isinstance(event.get("source_counts"), dict)
                    else {},
                    "max_context_block_tokens": event.get("max_context_block_tokens", 0),
                    "strategy": event.get("strategy", ""),
                }
                drained.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
                continue
            if stage == "context_packed":
                payload = {
                    "type": "context_packed",
                    "node": event.get("node_name", ""),
                    "llm_node": event.get("llm_node", ""),
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
                    "reason": event.get("reason", ""),
                    "warning": event.get("warning", ""),
                    "selected_tokens": event.get("selected_tokens"),
                    "budget_tokens": event.get("budget_tokens"),
                    "error_type": event.get("error_type", ""),
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

                    if event_type == "on_chain_end" and node_name == "mindmap_output":
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict) and output.get("mindmap_artifact"):
                            artifact = output["mindmap_artifact"]
                            mindmap_payload = json.dumps(
                                {
                                    "type": "mindmap_result",
                                    "title": artifact.get("title", "Markdown复习文档"),
                                    "tree": artifact.get("tree", {}),
                                    "xmind_url": artifact.get("xmind_url", ""),
                                },
                                ensure_ascii=False,
                            )
                            yield f"data: {mindmap_payload}\n\n"

                    if (
                        event_type == "on_chain_end"
                        and node_name == "review_doc_output"
                    ):
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict) and output.get(
                            "review_doc_artifact"
                        ):
                            artifact = output["review_doc_artifact"]
                            review_doc_payload = json.dumps(
                                {
                                    "type": "review_doc_result",
                                    "title": artifact.get("title", "Markdown复习文档"),
                                    "filename": artifact.get("filename", ""),
                                    "docx_filename": artifact.get("docx_filename", ""),
                                    "markdown_url": artifact.get("markdown_url", ""),
                                    "docx_url": artifact.get("docx_url", ""),
                                },
                                ensure_ascii=False,
                            )
                            yield f"data: {review_doc_payload}\n\n"
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
        await _try_update_run_state(
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
                    await _update_run_state(
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
                    return
                if (
                    isinstance(interrupt_value, dict)
                    and interrupt_value.get("type") == "memory_confirmation"
                ):
                    await _try_update_run_state(
                        graph,
                        config,
                        {
                            "resume_available": False,
                            "pending_interrupt_type": "memory_confirmation",
                        },
                        state=final_state,
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
                else:
                    await _try_update_run_state(
                        graph,
                        config,
                        {
                            "resume_available": False,
                            "pending_interrupt_type": "plan_review",
                        },
                        state=final_state,
                    )
                    payload_data = {
                        "type": "interrupt",
                        "interrupt_type": "plan_review",
                        "draft": interrupt_value,
                        "thread_id": thread_id,
                    }
                payload = json.dumps(payload_data, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                return

    await _try_update_run_state(
        graph,
        config,
        {
            "run_status": RUN_STATUS_COMPLETED,
            "resume_available": False,
            "pending_interrupt_type": "",
            "current_node": "",
            "stop_requested": False,
        },
        state=final_state,
    )
    run_control_registry.clear_stop_signal(thread_id)
    yield f"data: {json.dumps({'type': 'run_status', 'run_status': RUN_STATUS_COMPLETED, 'thread_id': thread_id}, ensure_ascii=False)}\n\n"

    resource_payload = _resource_final_payload(final_state)
    if resource_payload:
        # TEMP A3_TRACE: remove after resource final payload validation.
        emit_a3_trace(
            logger,
            "sse_resource_final",
            {
                "sent": True,
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

    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


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
    if user_id:
        try:
            manager = get_profile_manager()
            profile_ctx = await manager.build_profile_context(user_id)
            if profile_ctx:
                messages.insert(0, SystemMessage(content=profile_ctx))
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
    _emit_graph_config_trace(graph, config, state_input)

    yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'type': 'run_status', 'run_status': RUN_STATUS_RUNNING, 'thread_id': thread_id}, ensure_ascii=False)}\n\n"

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
        graph, state_input, config, thread_id, preserve_context_history=False
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

    if memory_use_choice:
        resume_value = {"type": "memory_confirmation", "choice": memory_use_choice}
    elif feedback:
        resume_value = {"action": "feedback", "text": feedback}
    else:
        resume_value = edited_plan

    resume_input = Command(resume=resume_value)
    _emit_graph_config_trace(
        graph,
        config,
        {
            "request_id": str(uuid.uuid4()),
            "session_id": thread_id,
            "thread_id": thread_id,
        },
    )

    yield f"data: {json.dumps({'type': 'run_status', 'run_status': RUN_STATUS_CONTINUING, 'thread_id': thread_id}, ensure_ascii=False)}\n\n"

    async for chunk in _stream_graph_events(
        graph, resume_input, config, thread_id, preserve_context_history=True
    ):
        yield chunk


async def get_thread_status_payload(graph, thread_id: str) -> ThreadStatusResponse:
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
    yield f"data: {json.dumps({'type': 'run_status', 'run_status': RUN_STATUS_CONTINUING, 'thread_id': thread_id}, ensure_ascii=False)}\n\n"

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
    return StreamingResponse(
        generate_resume_sse(
            req.edited_plan,
            req.feedback,
            request.app.state.graph,
            req.thread_id,
            memory_use_choice=req.memory_use_choice,
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
                content=f"鐢ㄦ埛鑷堪骞寸骇: {req.grade}",
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
        "Onboarding 鐢诲儚宸插垱寤?user=%s nickname=%s subjects=%d goals=%d",
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
