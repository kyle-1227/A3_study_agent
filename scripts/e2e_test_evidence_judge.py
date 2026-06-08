"""E2E test: evidence judge with openai/gpt-oss-120b:free"""
import os, sys, asyncio, json, logging
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

os.environ["LOG_A3_TRACE"] = "1"
os.environ["LOG_WEB_SEARCH_RESULT"] = "1"
os.environ["LOG_CONTEXT_ASSEMBLY"] = "1"
os.environ["LOG_GENERATION_SUMMARY"] = "1"
os.environ["LOG_RAG_RESULT"] = "1"
os.environ["LOG_RETRY_TRACE"] = "1"

logging.basicConfig(level=logging.WARNING, format="%(name)s [%(levelname)s] %(message)s")
logging.getLogger("src").setLevel(logging.WARNING)

from langchain_core.messages import HumanMessage
from src.graph.builder import get_compiled_graph


async def test():
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": "e2e-test-python-006"}}
    query = "总结一下python知识"

    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    final_state = None
    async for event in graph.astream(
        {"messages": [HumanMessage(content=query)]},
        config,
        stream_mode="values",
    ):
        final_state = event

    if not final_state:
        print("ERROR: No final state")
        return

    ej = final_state.get("evidence_judge_output", {})
    if isinstance(ej, dict):
        print("\n=== Evidence Judge Result ===")
        print(f"success: {ej.get('success')}")
        print(f"failure_phase: {ej.get('failure_phase', 'N/A')}")
        print(f"output_mode: {ej.get('output_mode', 'N/A')}")
        print(f"model: {ej.get('model', 'N/A')}")
        print(f"status_code: {ej.get('status_code', 'N/A')}")
        print(f"input_candidate_count: {ej.get('input_candidate_count', ej.get('candidate_count', 'N/A'))}")
        print(f"kept_count: {ej.get('kept_count', 'N/A')}")
        print(f"rejected_count: {ej.get('rejected_count', 'N/A')}")
        print(f"error_type: {ej.get('error_type', 'N/A')}")
        print(f"parsing_error: {ej.get('parsing_error', 'N/A')}")
        ve = ej.get("validation_error", "")
        print(f"validation_error: {str(ve)[:500]}")
        print(f"overall_evidence_state: {ej.get('overall_evidence_state', 'N/A')}")
        print(f"need_more_web_search: {ej.get('need_more_web_search', 'N/A')}")

    ctx = final_state.get("context", [])
    print(f"\n=== Context Assembly ===")
    print(f"context_count: {len(ctx)}")
    for i, doc in enumerate(ctx[:5]):
        print(f"  [{i}] id={doc.get('evidence_id', '?')}, source_type={doc.get('source_type', '?')}, subject={doc.get('subject', '?')}")
    print(f"evidence_judge_failed: {final_state.get('evidence_judge_failed')}")
    print(f"degraded_generation: {final_state.get('degraded_generation')}")

    msgs = final_state.get("messages", [])
    if msgs:
        last = msgs[-1]
        content = getattr(last, "content", str(last))
        is_blocked = "[开发诊断]" in content
        print(f"\n=== Final Output ===")
        if is_blocked:
            print("[BLOCKED] Generation blocked due to evidence judge failure")
            print(content[:1000])
        else:
            print(content[:2000])


if __name__ == "__main__":
    asyncio.run(test())
