"""Emotional response node — single LLM call with homeroom teacher persona."""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from src.config import get_setting, load_prompt
from src.graph.llm import invoke_plain_llm_fail_fast
from src.graph.state import LearningState
from src.tracing import traced_llm_call, traced_node


@traced_node
async def emotional_response(state: LearningState) -> dict:
    """Respond with warm, practical emotional support."""
    history = [SystemMessage(content=load_prompt("emotional_system"))]
    for msg in state["messages"]:
        history.append(msg)

    temperature = get_setting("emotional.temperature", 0.8)

    with traced_llm_call(
        model_name=get_setting("llm.emotional.model", ""),
        node_name="emotional_response",
        temperature=temperature,
    ):
        response = await invoke_plain_llm_fail_fast(
            node_name="emotional_response",
            llm_node="emotional",
            messages=history,
            state=state,
            temperature=temperature,
        )

    return {"messages": [AIMessage(content=response)]}
