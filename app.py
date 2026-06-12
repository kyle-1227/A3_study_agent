"""A3 Study Agent 鈥?AI-powered university learning resource generation system."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv(Path(__file__).parent / ".env")

from src.database.checkpointer import (
    checkpointer_enabled,
    checkpointer_type,
    get_db_uri,
    make_thread_config,
)
from src.graph.exercises import _render_exercise_markdown
from src.graph.builder import get_compiled_graph
from src.graph.state import CONTEXT_CLEAR, initial_request_reset_transient_state
from src.schemas import ChatRequest, ResumeRequest
from src.observability.a3_trace import emit_a3_trace
from src.tools.document_tool import get_review_doc_artifact_dir
from src.tools.mindmap_tool import get_mindmap_artifact_dir
from src.tracing import setup_tracing, shutdown_tracing

logger = logging.getLogger(__name__)


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
    resource_type = str(final_state.get("requested_resource_type") or "")
    mindmap_artifact = final_state.get("mindmap_artifact") or {}
    mindmap_tree = final_state.get("mindmap_tree") or {}
    exercise_items = final_state.get("exercise_items") or []
    exercise_artifact = final_state.get("exercise_artifact") or {}
    review_doc_artifact = final_state.get("review_doc_artifact") or {}
    study_plan_artifact = final_state.get("study_plan_artifact") or {}
    study_plan_document = final_state.get("study_plan_document_artifact") or {}

    if resource_type not in {"mindmap", "quiz", "review_doc", "study_plan"}:
        if mindmap_artifact or mindmap_tree:
            resource_type = "mindmap"
        elif exercise_items or exercise_artifact:
            resource_type = "quiz"
        elif review_doc_artifact:
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
            "title": review_doc_artifact.get("title", "Markdown Review Document"),
            "filename": review_doc_artifact.get("filename", ""),
            "docx_filename": review_doc_artifact.get("docx_filename", ""),
            "markdown_url": review_doc_artifact.get("markdown_url", ""),
            "docx_url": review_doc_artifact.get("docx_url", ""),
        }

    if resource_type == "study_plan" and (study_plan_artifact or study_plan_document):
        payload["study_plan"] = {
            "title": study_plan_artifact.get("title") or study_plan_document.get("title", "Personalized Study Plan"),
            "filename": study_plan_document.get("filename", ""),
            "docx_filename": study_plan_document.get("docx_filename", ""),
            "markdown_url": study_plan_document.get("markdown_url", ""),
            "docx_url": study_plan_document.get("docx_url", ""),
        }

    return payload



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

    try:
        async for event in graph.astream_events(input_data, config=config, version="v2"):
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
            "exercise_items_count": len(final_state.get("exercise_items") or []),
            "requested_resource_type": final_state.get("requested_resource_type", ""),
        },
        state=final_state,
        env_flag="LOG_A3_TRACE",
    )

    if state_snapshot.next:
        for task in state_snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                draft = task.interrupts[0].value
                payload = json.dumps(
                    {"type": "interrupt", "draft": draft, "thread_id": thread_id},
                    ensure_ascii=False,
                )
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
                "exercise_items_count": len(resource_payload.get("exercise_items") or []),
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
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    config = make_thread_config(thread_id)
    state_input = {
        "messages": [HumanMessage(content=query)],
        "request_id": request_id,
        "session_id": thread_id,
        "thread_id": thread_id,
        "context": CONTEXT_CLEAR,
        **initial_request_reset_transient_state(),
    }
    _emit_graph_config_trace(graph, config, state_input)

    # Emit thread_id so frontend can use it for /resume
    yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id}, ensure_ascii=False)}\n\n"

    async for chunk in _stream_graph_events(graph, state_input, config, thread_id):
        yield chunk


async def generate_resume_sse(
    edited_plan: str,
    feedback: str | None,
    graph,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """Resume an interrupted graph and stream remaining events as SSE.

    Args:
        edited_plan: The user-edited plan text to resume with.
        feedback: Optional feedback text for AI-driven plan revision.
        graph: The compiled LangGraph instance from app.state.
        thread_id: Session ID identifying the interrupted graph state.
    """
    config = make_thread_config(thread_id)

    if feedback:
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
        generate_sse(chat.query, request.app.state.graph, thread_id=chat.thread_id),
        media_type="text/event-stream",
    )


@app.post("/resume")
async def resume_endpoint(req: ResumeRequest, request: Request):
    return StreamingResponse(
        generate_resume_sse(req.edited_plan, req.feedback, request.app.state.graph, req.thread_id),
        media_type="text/event-stream",
    )


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
