from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.structured_output import ALLOWED_OUTPUT_MODES, invoke_structured_llm  # noqa: E402


class ProbeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["ok", "failed"] = "ok"
    answer: str = Field(..., description="A short answer proving structured output worked.")
    confidence: float = Field(..., ge=0, le=1)


def _messages() -> list:
    return [
        SystemMessage(
            content=(
                "Return only structured data for the requested schema. "
                "Do not include Markdown or extra prose."
            )
        ),
        HumanMessage(content="Return verdict ok, answer 'structured output works', confidence 0.9."),
    ]


async def _run(mode: str, fallback_modes: list[str]) -> None:
    result = await invoke_structured_llm(
        node_name="evidence_judge",
        schema=ProbeOutput,
        messages=_messages(),
        output_mode=mode,
        fallback_modes=fallback_modes,
        max_raw_chars=12000,
    )
    payload = {
        "node_name": result.node_name,
        "provider": result.provider,
        "model": result.model,
        "output_mode": result.output_mode,
        "fallback_modes": result.fallback_modes,
        "success": result.success,
        "failure_phase": result.failure_phase,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "status_code": result.status_code,
        "provider_error_body": result.provider_error_body,
        "parsed": result.parsed.model_dump(mode="json") if result.parsed else None,
        "raw_output": result.raw_output,
        "attempts": [attempt.__dict__ for attempt in result.attempts],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Probe provider-neutral structured output modes.")
    parser.add_argument(
        "--mode",
        default="native_json_schema_pydantic",
        choices=sorted(ALLOWED_OUTPUT_MODES),
        help="Provider-neutral structured output mode to test.",
    )
    parser.add_argument(
        "--fallback-mode",
        action="append",
        default=[],
        choices=sorted(ALLOWED_OUTPUT_MODES),
        help="Optional explicit fallback mode. May be passed multiple times.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.mode, args.fallback_mode))


if __name__ == "__main__":
    main()

