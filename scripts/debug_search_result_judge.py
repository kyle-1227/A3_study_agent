"""Debug OpenRouter Search Result Judge against sample or saved Tavily results."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.graph.academic import (  # noqa: E402
    _build_judge_messages,
    _judge_request_payload,
    _judge_tavily_search_results_with_llm,
)


def _sample_results() -> list[dict]:
    return [
        {
            "title": "[PDF] Chapter 6 Eigenvalues and Eigenvectors",
            "url": "https://math.mit.edu/~gs/linearalgebra/ila5/linearalgebra5_6-1.pdf",
            "content": "MIT linear algebra textbook chapter PDF covering eigenvalues, eigenvectors, examples, and exercises.",
            "score": 0.93,
        },
        {
            "title": "Linear Algebra Practice Problems with Solutions",
            "url": "https://example.edu/course/linear-algebra/problem-set-solutions",
            "content": "Open course problem set with matrices, determinants, eigenvalues, and worked solutions.",
            "score": 0.85,
        },
        {
            "title": "Ultimate Linear Algebra Trivia Quiz",
            "url": "https://example-quiz-site.test/ultimate-linear-algebra-trivia",
            "content": "Short trivia questions with little explanation and no academic source context.",
            "score": 0.52,
        },
        {
            "title": "Linear Algebra Explained - YouTube",
            "url": "https://www.youtube.com/watch?v=example",
            "content": "Video introduction to vectors, matrices, and eigenvalues.",
            "score": 0.48,
        },
    ]


def _load_results(path: str | None) -> list[dict]:
    if not path:
        return _sample_results()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("results"), list):
        return raw["results"]
    if isinstance(raw, list):
        return raw
    raise ValueError("JSON file must contain a list or an object with a results list")


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Debug OpenRouter Search Result Judge.")
    parser.add_argument("--results-json", help="Path to saved Tavily results JSON")
    parser.add_argument("--original", default="给我一份高等数学的习题，可以扩展到线性代数练习")
    parser.add_argument("--query", default="linear algebra practice problems matrices eigenvalues eigenvectors")
    parser.add_argument("--raw-query", default="")
    parser.add_argument("--subject", default="math")
    parser.add_argument("--role", default="core_concept")
    parser.add_argument("--purpose", default="resource_enrichment")
    parser.add_argument("--coverage-risk", default="medium")
    parser.add_argument("--local-evidence-strength", default="usable")
    args = parser.parse_args()

    results = _load_results(args.results_json)
    raw_query = args.raw_query or args.query
    state = {
        "messages": [HumanMessage(content=args.original)],
        "learning_goal": args.original,
        "requested_resource_type": "quiz",
    }
    messages = _build_judge_messages(
        state=state,
        subject=args.subject,
        role=args.role,
        purpose=args.purpose,
        search_query=args.query,
        raw_query=raw_query,
        original_user_query=args.original,
        tavily_results=results,
        coverage_risk=args.coverage_risk,
        local_evidence_strength=args.local_evidence_strength,
    )
    request_payload = _judge_request_payload(messages)
    accepted, debug = await _judge_tavily_search_results_with_llm(
        state=state,
        subject=args.subject,
        role=args.role,
        purpose=args.purpose,
        search_query=args.query,
        raw_query=raw_query,
        original_user_query=args.original,
        tavily_results=results,
        coverage_risk=args.coverage_risk,
        local_evidence_strength=args.local_evidence_strength,
    )

    report = {
        "provider": debug.get("provider"),
        "model": debug.get("model"),
        "success": debug.get("success"),
        "accepted_count": len(accepted),
        "rejected_count": debug.get("rejected_count"),
        "failure_phase": debug.get("failure_phase", ""),
        "error_type": debug.get("error_type", ""),
        "error_message": debug.get("error_message", ""),
        "request_payload_preview": {
            "model": request_payload.get("model"),
            "temperature": request_payload.get("temperature"),
            "max_tokens": request_payload.get("max_tokens"),
            "response_format": request_payload.get("response_format"),
            "messages": [
                {
                    "role": item.get("role"),
                    "content_preview": str(item.get("content", ""))[:2000],
                }
                for item in request_payload.get("messages", [])
            ],
        },
        "raw_output": debug.get("raw_output") or debug.get("raw_preview", ""),
        "parsed": debug.get("judged_results", []),
        "parsing_error": debug.get("parsing_error", ""),
        "validation_error": debug.get("validation_error", ""),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if debug.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
