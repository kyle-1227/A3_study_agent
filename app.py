"""A3 Study Agent 鈥?AI-powered university learning resource generation system."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

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
from src.graph.state import CONTEXT_CLEAR, MEMORY_CLEAR, initial_request_reset_transient_state
from src.profile import get_profile_manager
from src.schemas import ChatRequest, OnboardRequest, ProfileResponse, ResumeRequest
from src.observability.a3_trace import emit_a3_trace, reset_trace_event_sink, set_trace_event_sink
from src.tools.document_tool import get_exercise_artifact_dir, get_review_doc_artifact_dir
from src.tools.mindmap_tool import get_mindmap_artifact_dir
from src.tracing import setup_tracing, shutdown_tracing

logger = logging.getLogger(__name__)
PROVIDER_RETRY_TRACE_STAGES = {
    "provider_transport_retry_attempt",
    "provider_transport_error",
    "final_failure_after_retries",
}


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
                logger.warning("DB_URI is set but PostgreSQL checkpointer was unavailable; using MemorySaver")
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
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ALLOWED_NODES = {"generate_answer", "emotional_response"}

# Non-streaming nodes whose final AIMessage content is emitted as a "text" SSE event.
TEXT_EMIT_NODES = {"handle_unknown", "evidence_summary_output", "mindmap_output", "exercise_output", "review_doc_output", "study_plan_output"}

# All graph nodes whose lifecycle (start/end) we broadcast to the frontend.
GRAPH_NODES = {
    "supervisor",
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


def _last_ai_message_content(final_state: dict) -> str:
    for msg in reversed(final_state.get("messages") or []):
        content = getattr(msg, "content", "")
        if content:
            return str(content)
    return ""


def _resource_final_payload(final_state: dict) -> dict | None:
    if final_state.get("evidence_controlled_stop") is True or final_state.get("final_response_type") == "evidence_summary":
        answer = _last_ai_message_content(final_state) or str(final_state.get("plan") or "")
        return {
            "type": "resource_final",
            "resource_type": "evidence_summary",
            "controlled_stop": True,
            "controlled_stop_reason": final_state.get("evidence_controlled_stop_reason", ""),
            "answer": answer,
        }

    resource_type = str(final_state.get("requested_resource_type") or "")
    mindmap_artifact = final_state.get("mindmap_artifact") or {}
    mindmap_tree = final_state.get("mindmap_tree") or {}
    exercise_items = final_state.get("exercise_items") or []
    exercise_artifact = final_state.get("exercise_artifact") or {}
    review_doc_artifact = final_state.get("review_doc_artifact") or {}
    review_doc_artifacts = final_state.get("review_doc_artifacts") or []
    study_plan_artifact = final_state.get("study_plan_artifact") or {}
    study_plan_document = final_state.get("study_plan_document_artifact") or {}

    if resource_type not in {"mindmap", "quiz", "review_doc", "study_plan"}:
        if mindmap_artifact or mindmap_tree:
            resource_type = "mindmap"
        elif exercise_items or exercise_artifact:
            resource_type = "quiz"
        elif review_doc_artifact or review_doc_artifacts:
            resource_type = "review_doc"
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

    if resource_type == "mindmap" and (mindmap_artifact or mindmap_tree):
        payload["mindmap"] = {
            "title": mindmap_artifact.get("title", "Knowledge Mindmap"),
            "tree": (mindmap_artifact.get("tree") or mindmap_tree or {}),
            "xmind_url": mindmap_artifact.get("xmind_url", ""),
        }

    if resource_type == "quiz":
        if (not answer or len(answer.strip()) < 40) and exercise_items:
            title = str(exercise_artifact.get("title") or "Leveled exercises")
            answer = _render_exercise_markdown(
                title,
                exercise_items,
                review_reason=str(exercise_artifact.get("review_reason") or final_state.get("exercise_review_reason") or ""),
                quality_warning=bool(exercise_artifact.get("quality_warning")),
            )
            payload["answer"] = answer
        payload["exercise_items"] = exercise_items
        payload["exercise_artifact"] = exercise_artifact

    if resource_type == "review_doc" and review_doc_artifact:
        payload["review_doc"] = {
            "subject": review_doc_artifact.get("subject", ""),
            "title": review_doc_artifact.get("title", "Markdown Review Document"),
            "filename": review_doc_artifact.get("filename", ""),
            "docx_filename": review_doc_artifact.get("docx_filename", ""),
            "markdown_url": review_doc_artifact.get("markdown_url", ""),
            "docx_url": review_doc_artifact.get("docx_url", ""),
            "markdown": review_doc_artifact.get("markdown", ""),
        }
    if resource_type == "review_doc" and review_doc_artifacts:
        payload["review_doc_artifacts"] = [
            {
                "subject": artifact.get("subject", ""),
                "title": artifact.get("title", "Markdown Review Document"),
                "filename": artifact.get("filename", ""),
                "docx_filename": artifact.get("docx_filename", ""),
                "markdown_url": artifact.get("markdown_url", ""),
                "docx_url": artifact.get("docx_url", ""),
                "markdown": artifact.get("markdown", ""),
            }
            for artifact in review_doc_artifacts
        ]

    if resource_type == "study_plan" and (study_plan_artifact or study_plan_document):
        payload["study_plan"] = {
            "title": study_plan_artifact.get("title") or study_plan_document.get("title", "Personalized Study Plan"),
            "filename": study_plan_document.get("filename", ""),
            "docx_filename": study_plan_document.get("docx_filename", ""),
            "markdown_url": study_plan_document.get("markdown_url", ""),
            "docx_url": study_plan_document.get("docx_url", ""),
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
    ]
    values = {
        "conversation_summary": "",
        "evidence_summary_memory": MEMORY_CLEAR,
        "evidence_gap_memory": MEMORY_CLEAR,
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
) -> AsyncGenerator[str, None]:
    """Shared SSE event streaming logic for /stream and /resume.

    Processes astream_events and yields SSE payloads for node lifecycle,
    token streaming, usage, and interrupt events.
    """
    node_start_times: dict[str, float] = {}
    active_nodes: list[str] = []
    trace_events: list[dict] = []
    trace_sink_token = set_trace_event_sink(trace_events)

    def _drain_provider_retry_events() -> list[str]:
        drained: list[str] = []
        while trace_events:
            event = trace_events.pop(0)
            if event.get("stage") not in PROVIDER_RETRY_TRACE_STAGES:
                continue
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
        return drained

    try:
        async for event in graph.astream_events(input_data, config=config, version="v2"):
            for retry_payload in _drain_provider_retry_events():
                yield retry_payload
            event_type = event["event"]

            # 鈹€鈹€ Node lifecycle events 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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
                        payload = json.dumps(
                            {"type": "node_event", "status": "start", "node": node_name},
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
                    yield f"data: {payload}\n\n"

                    # Emit "text" for non-streaming nodes (AC-02)
                    if event_type == "on_chain_end" and node_name in TEXT_EMIT_NODES:
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict):
                            for msg in output.get("messages", []):
                                if hasattr(msg, "content") and msg.content:
                                    text_payload = json.dumps(
                                        {"type": "text", "content": msg.content, "node": node_name},
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
                                    "title": artifact.get("title", "鐭ヨ瘑鐐规€濈淮瀵煎浘"),
                                    "tree": artifact.get("tree", {}),
                                    "xmind_url": artifact.get("xmind_url", ""),
                                },
                                ensure_ascii=False,
                            )
                            yield f"data: {mindmap_payload}\n\n"

                    if event_type == "on_chain_end" and node_name == "review_doc_output":
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict) and output.get("review_doc_artifact"):
                            artifact = output["review_doc_artifact"]
                            review_doc_payload = json.dumps(
                                {
                                    "type": "review_doc_result",
                                    "title": artifact.get("title", "Markdown澶嶄範鏂囨。"),
                                    "filename": artifact.get("filename", ""),
                                    "docx_filename": artifact.get("docx_filename", ""),
                                    "markdown_url": artifact.get("markdown_url", ""),
                                    "docx_url": artifact.get("docx_url", ""),
                                },
                                ensure_ascii=False,
                            )
                            yield f"data: {review_doc_payload}\n\n"
                    for retry_payload in _drain_provider_retry_events():
                        yield retry_payload

            # 鈹€鈹€ Token streaming 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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

            # 鈹€鈹€ Token usage events 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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
    except Exception as e:
        for retry_payload in _drain_provider_retry_events():
            yield retry_payload
        logger.exception("Unhandled error in graph streaming")
        failed_node = active_nodes[-1] if active_nodes else None
        if failed_node:
            start_t = node_start_times.get(failed_node)
            duration_ms = round((time.monotonic() - start_t) * 1000) if start_t is not None else None
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
            {"type": "error", "message": str(e), "failed_node": failed_node, "active_nodes": active_nodes},
            ensure_ascii=False,
        )
        yield f"data: {error_payload}\n\n"
        return
    finally:
        reset_trace_event_sink(trace_sink_token)

    # 鈹€鈹€ Check for interrupt after stream completes 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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
            "review_doc_artifacts_count": len(final_state.get("review_doc_artifacts") or []),
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
                if isinstance(interrupt_value, dict) and interrupt_value.get("type") == "memory_confirmation":
                    payload_data = {
                        "type": "interrupt",
                        "interrupt_type": "memory_confirmation",
                        "question": interrupt_value.get("question", ""),
                        "reason": interrupt_value.get("reason", ""),
                        "selected_memory_count": interrupt_value.get("selected_memory_count", 0),
                        "options": interrupt_value.get("options", []),
                        "thread_id": thread_id,
                    }
                else:
                    payload_data = {
                        "type": "interrupt",
                        "interrupt_type": "plan_review",
                        "draft": interrupt_value,
                        "thread_id": thread_id,
                    }
                payload = json.dumps(payload_data, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                return

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
                "review_doc_artifacts_count": len(resource_payload.get("review_doc_artifacts") or []),
                "has_review_doc_artifacts": bool(resource_payload.get("review_doc_artifacts")),
                "exercise_items_count": len(resource_payload.get("exercise_items") or []),
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
      鈥?emitted once at stream start so frontend can use it for /resume.
    * ``{"type": "node_event", "status": "start"|"end", "node": "<name>"}``
      鈥?emitted when a graph node begins or finishes execution.
    * ``{"type": "token", "content": "<text>"}``
      鈥?emitted for each streamed token from an allowed LLM node.
    * ``{"type": "interrupt", "draft": "...", "thread_id": "..."}``
      鈥?emitted when the graph pauses for human review (HIL).

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

    # Inject profile context as a SystemMessage if a user profile exists
    messages = [HumanMessage(content=query)]
    if user_id:
        try:
            manager = get_profile_manager()
            profile_ctx = await manager.build_profile_context(user_id)
            if profile_ctx:
                messages.insert(0, SystemMessage(content=profile_ctx))
                logger.info("注入画像上下文 user=%s (%d chars)", user_id, len(profile_ctx))
        except Exception:
            logger.exception("获取画像上下文失败 user=%s", user_id)

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

    async for chunk in _stream_graph_events(graph, state_input, config, thread_id):
        yield chunk

    # Record the conversation turn for profile evolution (non-fatal)
    if user_id:
        try:
            await manager.process_conversation(
                user_id=user_id,
                user_message=query,
                assistant_response="",
            )
            logger.debug("画像轮次已记录 user=%s", user_id)
        except Exception:
            logger.exception("画像记录失败（非致命） user=%s", user_id)


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
        {"request_id": str(uuid.uuid4()), "session_id": thread_id, "thread_id": thread_id},
    )

    async for chunk in _stream_graph_events(graph, resume_input, config, thread_id):
        yield chunk


@app.post("/stream")
async def stream_endpoint(chat: ChatRequest, request: Request):
    return StreamingResponse(
        generate_sse(chat.query, request.app.state.graph, thread_id=chat.thread_id, user_id=chat.user_id),
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

    if not artifact_path.is_file() or artifact_path.suffix.lower() not in {".md", ".docx"}:
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

    if not artifact_path.is_file() or artifact_path.suffix.lower() not in {".md", ".docx"}:
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


# ── User profile & onboarding ────────────────────────────────────────────────


@app.post("/onboard")
async def onboard_endpoint(req: OnboardRequest):
    """Create an initial user profile from the onboarding wizard.

    All values are explicit self-reports → stored with confidence=0.9.
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
        obs_list.append(AgentObservation(
            content=f"用户自述年级: {req.grade}",
            category="general",
            importance=0.8,
            created_at=now,
        ))
    if req.subjects:
        obs_list.append(AgentObservation(
            content=f"用户首次选择 {len(req.subjects)} 个学习方向: {', '.join(req.subjects)}",
            category="general",
            importance=0.8,
            created_at=now,
        ))

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
        "Onboarding 画像已创建 user=%s nickname=%s subjects=%d goals=%d",
        req.user_id, req.nickname, len(skills), len(goals),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
