"""Gaokao Tutor — AI-powered tutoring assistant for Chinese Gaokao preparation."""

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
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv(Path(__file__).parent / ".env")

from src.database.checkpointer import get_db_uri, make_thread_config
from src.graph.builder import get_compiled_graph
from src.schemas import ChatRequest, ResumeRequest
from src.tracing import setup_tracing, shutdown_tracing

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage async resources: tracing, PostgreSQL checkpointer, graph."""
    setup_tracing()

    async with AsyncExitStack() as stack:
        checkpointer = None
        db_uri = get_db_uri()

        if db_uri:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                checkpointer = await stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(db_uri)
                )
                await checkpointer.setup()
                logger.info("PostgreSQL checkpointer initialized")
            except Exception:
                logger.exception(
                    "Failed to initialize PostgreSQL checkpointer, running stateless"
                )
                checkpointer = None
        else:
            logger.info("DB_URI not set, running without persistent state")

        app.state.graph = get_compiled_graph(checkpointer=checkpointer)
        yield

    shutdown_tracing()


app = FastAPI(title="Gaokao Tutor API", lifespan=lifespan)

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

FastAPIInstrumentor.instrument_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ALLOWED_NODES = {"generate_answer", "plan_adversarial", "emotional_response"}

# All graph nodes whose lifecycle (start/end) we broadcast to the frontend.
GRAPH_NODES = {
    "supervisor",
    "academic_router",
    "rag_retrieve",
    "web_search",
    "generate_answer",
    "evaluate_hallucination",
    "search_policy",
    "gather_intel",
    "plan_adversarial",
    "emotional_response",
    "handle_unknown",
}


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

    async for event in graph.astream_events(input_data, config=config, version="v2"):
        event_type = event["event"]

        # ── Node lifecycle events ──────────────────────────────────────
        if event_type in ("on_chain_start", "on_chain_end"):
            node_name = event.get("name")
            meta_node = event.get("metadata", {}).get("langgraph_node")
            # Only emit for top-level graph nodes (name matches metadata),
            # not for internal sub-chains (RunnableSequence, etc.).
            if node_name and node_name == meta_node and node_name in GRAPH_NODES:
                if event_type == "on_chain_start":
                    node_start_times[node_name] = time.monotonic()
                    payload = json.dumps(
                        {"type": "node_event", "status": "start", "node": node_name},
                        ensure_ascii=False,
                    )
                else:
                    duration_ms = None
                    start_t = node_start_times.pop(node_name, None)
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

        # ── Token streaming ────────────────────────────────────────────
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

        # ── Token usage events ─────────────────────────────────────────
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

    # ── Check for interrupt after stream completes ─────────────────
    state_snapshot = await graph.aget_state(config)
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


async def generate_sse(
    query: str,
    graph,
    thread_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream LangGraph events as Server-Sent Events (SSE).

    Yields SSE payload types:

    * ``{"type": "thread_id", "thread_id": "..."}``
      — emitted once at stream start so frontend can use it for /resume.
    * ``{"type": "node_event", "status": "start"|"end", "node": "<name>"}``
      — emitted when a graph node begins or finishes execution.
    * ``{"type": "token", "content": "<text>"}``
      — emitted for each streamed token from an allowed LLM node.
    * ``{"type": "interrupt", "draft": "...", "thread_id": "..."}``
      — emitted when the graph pauses for human review (HIL).

    Args:
        query: The user-provided string to be processed by the graph.
        graph: The compiled LangGraph instance from app.state.
        thread_id: Optional session ID for multi-turn memory. Auto-generated if None.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = make_thread_config(thread_id)
    state_input = {"messages": [HumanMessage(content=query)]}

    # Emit thread_id so frontend can use it for /resume
    yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id}, ensure_ascii=False)}\n\n"

    async for chunk in _stream_graph_events(graph, state_input, config, thread_id):
        yield chunk


async def generate_resume_sse(
    edited_plan: str,
    graph,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """Resume an interrupted graph and stream remaining events as SSE.

    Args:
        edited_plan: The user-edited plan text to resume with.
        graph: The compiled LangGraph instance from app.state.
        thread_id: Session ID identifying the interrupted graph state.
    """
    config = make_thread_config(thread_id)
    resume_input = Command(resume=edited_plan)

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
        generate_resume_sse(req.edited_plan, request.app.state.graph, req.thread_id),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
