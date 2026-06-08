from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Inlined schemas (avoid heavy src.graph import chain) ──────────────

class EvidenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., description="Stable internal id, e.g. local:math:0 or web:math:0")
    source_type: Literal["local_rag", "web"] = Field(...)

    provider: str = ""
    subject: str = ""
    role: str = ""
    purpose: str = ""

    title: str = ""
    source: str = ""
    url: str = ""
    content_preview: str = ""

    raw_vector_score: float | None = None
    raw_vector_score_source: str | None = None
    raw_vector_score_direction: str | None = None
    rerank_score: float | None = None
    branch_status: str | None = None
    branch_status_score_source: str | None = None

    tavily_score: float | None = None
    tavily_query: str | None = None

    metadata: dict = Field(default_factory=dict)


class EvidenceJudgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., description="Must match input EvidenceCandidate.evidence_id")
    keep: bool = Field(...)

    final_quality: Literal["high", "medium", "low"] = "low"
    relevance: Literal["high", "medium", "low"] = "low"
    authority: Literal["high", "medium", "low"] = "low"
    usefulness: Literal["high", "medium", "low"] = "low"
    risk: Literal["high", "medium", "low"] = "low"

    evidence_type: Literal[
        "local_course_material",
        "local_textbook_chunk",
        "local_exercise_answer",
        "university_course_pdf",
        "textbook_or_notes",
        "official_documentation",
        "open_exercise_set",
        "github_or_notebook",
        "educational_platform",
        "document_sharing_platform",
        "commercial_study_site",
        "video",
        "blog_or_article",
        "unknown",
    ] = "unknown"

    use_case: Literal[
        "core_evidence",
        "exercise_material",
        "implementation_reference",
        "background_context",
        "tool_ecosystem",
        "latest_practice",
        "inspiration_only",
        "discard",
    ] = "discard"

    coverage_contribution: str = ""
    reason: str = Field(...)


class EvidenceCoverageGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = ""
    role: str = ""
    gap: str = Field(..., description="What important coverage is missing or weak.")
    suggested_search_query: str = Field(
        ...,
        description="Concise English-first Tavily search query for future search optimization.",
    )
    purpose: Literal[
        "coverage_expansion",
        "resource_enrichment",
        "application_context",
        "tool_ecosystem",
        "latest_practice",
        "case_example",
        "implementation_detail",
        "comparison",
        "planning_support",
    ] = "coverage_expansion"
    priority: float = 0.5


class EvidenceJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_evidence_state: Literal[
        "sufficient",
        "partially_sufficient",
        "insufficient",
    ] = "insufficient"

    need_more_web_search: bool = False
    judged_evidence: list[EvidenceJudgeItem] = Field(default_factory=list)
    coverage_gaps: list[EvidenceCoverageGap] = Field(default_factory=list)
    decision_summary: str = ""


class MinimalJudgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    keep: bool
    reason: str


class MinimalJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judged_evidence: list[MinimalJudgeItem] = Field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────

def _sanitize_error_message(raw: Any, max_chars: int = 2000) -> str:
    text = str(raw) if raw else ""
    redacted = text
    for secret in [os.getenv("OPENROUTER_API_KEY", ""), os.getenv("TAVILY_API_KEY", "")]:
        if secret and len(secret) > 8:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted[:max_chars]


def _openrouter_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    title = os.getenv("OPENROUTER_APP_TITLE", "A3 Study Agent").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _candidate(index: int, *, source_type: Literal["local_rag", "web"] = "local_rag") -> EvidenceCandidate:
    if source_type == "web":
        return EvidenceCandidate(
            evidence_id=f"web:big_data:0:{index}",
            source_type="web",
            provider="tavily",
            subject="big_data",
            role="core_concept",
            purpose="first_round_dual_source",
            title="Big Data Tools Overview",
            source="https://example.edu/big-data-tools",
            url="https://example.edu/big-data-tools",
            content_preview="Course-style overview of Hadoop, Spark, Flink, Kafka, data lakes, lakehouse architecture, and current big data tooling.",
            tavily_score=0.82,
            tavily_query="big data latest tools Hadoop Spark Flink Kafka lakehouse",
        )
    return EvidenceCandidate(
        evidence_id=f"local:big_data:{index}",
        source_type="local_rag",
        provider="chroma_rag",
        subject="big_data",
        role="core_concept",
        purpose="local_course_retrieval",
        title="big_data_course_notes.pdf",
        source="big_data_course_notes.pdf",
        content_preview="Local course notes about big data concepts, distributed storage, distributed computing, Hadoop ecosystem, Spark, Flink, Kafka, and exercises.",
        rerank_score=0.91,
        branch_status="strong",
        branch_status_score_source="rerank_score",
    )


def _build_candidates(count: int) -> list[dict[str, Any]]:
    values: list[EvidenceCandidate] = []
    for index in range(count):
        source_type: Literal["local_rag", "web"] = "local_rag" if index % 2 == 0 else "web"
        values.append(_candidate(index, source_type=source_type))
    return [item.model_dump(mode="json") for item in values]


def _case_config(case: str) -> tuple[type[BaseModel], int]:
    if case == "minimal_1":
        return MinimalJudgeOutput, 1
    if case == "full_1":
        return EvidenceJudgeOutput, 1
    if case == "full_6":
        return EvidenceJudgeOutput, 6
    if case == "full_16":
        return EvidenceJudgeOutput, 16
    raise ValueError(f"Unsupported probe case: {case}")


def _messages(candidates: list[dict[str, Any]], *, minimal: bool) -> list[dict[str, str]]:
    if minimal:
        task = (
            "Judge every candidate. Return JSON with judged_evidence. "
            "Each item must contain evidence_id, keep, reason."
        )
    else:
        task = (
            "You are an Evidence Judge. Judge every EvidenceCandidate exactly once. "
            "Do not answer the user. Return only strict JSON matching the schema."
        )
    return [
        {
            "role": "system",
            "content": "Return only valid JSON matching the provided json_schema. Do not output markdown.",
        },
        {
            "role": "user",
            "content": (
                f"{task}\n"
                "original_user_query: 帮我总结大数据知识，并给出前沿工具和问答题\n"
                f"evidence_candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
            ),
        },
    ]


def _payload(model: str, schema_model: type[BaseModel], messages: list[dict[str, str]]) -> dict[str, Any]:
    schema = schema_model.model_json_schema()
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1800,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_model.__name__,
                "strict": True,
                "schema": schema,
            },
        },
    }


def _extract_output(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") if isinstance(response_json, dict) else None
    if not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    if isinstance(content, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return str(content or "")


def _provider_name_from_error_body(raw_body: str) -> str:
    try:
        parsed = json.loads(raw_body)
    except Exception:
        return ""

    def _walk(value: Any) -> str:
        if isinstance(value, dict):
            for key in ("provider_name", "provider", "providerName"):
                if value.get(key):
                    return str(value.get(key))
            for nested in value.values():
                found = _walk(nested)
                if found:
                    return found
        if isinstance(value, list):
            for nested in value:
                found = _walk(nested)
                if found:
                    return found
        return ""

    return _walk(parsed)


def run_probe(model: str, case: str) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    provider = "openrouter"
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    schema_model, candidate_count = _case_config(case)
    candidates = _build_candidates(candidate_count)
    messages = _messages(candidates, minimal=schema_model is MinimalJudgeOutput)
    payload = _payload(model, schema_model, messages)
    schema_text = json.dumps(payload["response_format"]["json_schema"]["schema"], ensure_ascii=False)
    prompt_text = "\n".join(message["content"] for message in messages)
    diagnostics: dict[str, Any] = {
        "case": case,
        "provider": provider,
        "model": model,
        "output_mode": "native_json_schema",
        "candidate_count": candidate_count,
        "prompt_chars": len(prompt_text),
        "schema_name": schema_model.__name__,
        "schema_size_chars": len(schema_text),
        "success": False,
        "status_code": None,
        "provider_name": "",
        "error_type": "",
        "error_message": "",
        "raw_error_body": "",
        "raw_output": "",
        "parsing_error": "",
        "validation_error": "",
        "elapsed_ms": None,
    }
    if not api_key:
        diagnostics.update({
            "success": False,
            "error_type": "MissingApiKey",
            "error_message": "OPENROUTER_API_KEY is not configured",
        })
        return diagnostics
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                headers=_openrouter_headers(api_key),
                json=payload,
            )
        diagnostics["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        diagnostics["status_code"] = response.status_code
        if response.status_code >= 400:
            raw_body = response.text
            provider_name = _provider_name_from_error_body(raw_body)
            diagnostics.update({
                "success": False,
                "error_type": "HTTPStatusError",
                "error_message": _sanitize_error_message(raw_body, max_chars=2000),
                "raw_error_body": _sanitize_error_message(raw_body, max_chars=12000),
                "provider_name": provider_name,
            })
            return diagnostics
        response_json = response.json()
        raw_output = _extract_output(response_json)
        diagnostics["raw_output"] = raw_output[:12000]

        # Try to parse and validate
        try:
            parsed_json = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            diagnostics.update({
                "success": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "parsing_error": str(exc),
            })
            return diagnostics

        try:
            schema_model.model_validate(parsed_json)
        except ValidationError as exc:
            diagnostics.update({
                "success": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "validation_error": str(exc),
            })
            return diagnostics

        diagnostics.update({
            "success": True,
        })
        return diagnostics
    except Exception as exc:
        diagnostics["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        diagnostics.update({
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": _sanitize_error_message(exc, max_chars=2000),
        })
        return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe OpenRouter strict json_schema support for Evidence Judge.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to probe (default: from EVIDENCE_JUDGE_MODEL env or settings).",
    )
    parser.add_argument(
        "--case",
        choices=["minimal_1", "full_1", "full_6", "full_16"],
        default="minimal_1",
    )
    args = parser.parse_args()

    # Resolve model: CLI arg > env > settings.yaml > fallback
    if args.model:
        model = args.model
    else:
        # Try reading from the same sources as the production Evidence Judge
        try:
            from src.config.config_manager import get_setting
            model = str(get_setting("llm.evidence_judge.model", "") or "").strip()
        except Exception:
            model = ""
        if not model:
            model = os.getenv("EVIDENCE_JUDGE_MODEL", "").strip()
        if not model:
            model = "nvidia/nemotron-3-ultra-550b-a55b:free"

    result = run_probe(model, args.case)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
