"""Pure Web Research V2 schemas, validators, and URL helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field


WebEvidenceType = Literal[
    "university_course_page",
    "university_course_pdf",
    "official_documentation",
    "tutorial",
    "paper",
    "blog_article",
    "forum_discussion",
    "video",
    "unknown",
]
WebUseCase = Literal[
    "core_evidence",
    "implementation_reference",
    "exercise_material",
    "roadmap_reference",
    "background_context",
    "discard",
]
WebQuality = Literal["high", "medium", "low"]
WebRisk = Literal["low", "medium", "high"]
WebFetchStatus = Literal["success", "failed", "not_attempted"]


class WebResearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(...)
    subject: str = Field(...)
    role: str = Field(...)
    purpose: str = Field(...)
    search_query: str = Field(...)
    reason: str = Field(
        ...,
        description=(
            "Required for every web research task. Explain why this search task is needed "
            "for the user's current learning goal and what coverage it is expected to retrieve. "
            "This is not a keep/reject reason for a source."
        ),
    )
    priority: float = Field(...)


class WebResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[WebResearchTask] = Field(default_factory=list, max_length=6)


class WebSourceSummary(BaseModel):
    """LLM summary for a program-provided source id.

    URL, title, and domain are intentionally absent so the model cannot invent
    or mutate source identity. Those fields remain program/Tavily-derived.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(...)
    keep: bool = Field(...)
    summary: str = Field("")
    coverage_points: list[str] = Field(default_factory=list, max_length=8)
    reason: str = Field(...)
    evidence_type: WebEvidenceType = "unknown"
    use_case: WebUseCase = "discard"
    relevance: WebQuality = "low"
    usefulness: WebQuality = "low"
    risk: WebRisk = "medium"


class WebSourceSummaryBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summaries: list[WebSourceSummary] = Field(default_factory=list, max_length=12)


class WebSourceSummarizerSourceDTO(BaseModel):
    """Minimal LLM-facing input for one web source summary item."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(...)
    source_text: str = Field(...)
    source_context: str = Field(...)
    provider_score: float | None = None


class WebSourceSummarizerInputDTO(BaseModel):
    """LLM-facing input envelope for the Web Source Summarizer."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(...)
    requested_resource_type: str = ""
    requested_resource_types: list[str] = Field(default_factory=list)
    output_language: str = "same_as_user_query"
    sources: list[WebSourceSummarizerSourceDTO] = Field(default_factory=list)


def build_web_source_summarizer_input_dto(
    *,
    query: str,
    learning_goal: str,
    requested_resource_type: str,
    requested_resource_types: list[str],
    output_language: str,
    sources: list[dict[str, Any]],
    max_source_text_chars: int = 1600,
    max_source_context_chars: int = 600,
) -> WebSourceSummarizerInputDTO:
    """Project internal source records into the narrow LLM-facing DTO."""

    dto_sources: list[WebSourceSummarizerSourceDTO] = []
    for source in sources:
        source_id = str(source.get("source_id") or "")
        source_text = _summarizer_source_text(source, max_chars=max_source_text_chars)
        source_context = _summarizer_source_context(
            source,
            learning_goal=learning_goal,
            requested_resource_type=requested_resource_type,
            requested_resource_types=requested_resource_types,
            max_chars=max_source_context_chars,
        )
        provider_score = source.get("provider_score")
        try:
            provider_score = float(provider_score) if provider_score is not None else None
        except Exception:
            provider_score = None
        dto_sources.append(
            WebSourceSummarizerSourceDTO(
                source_id=source_id,
                source_text=source_text,
                source_context=source_context,
                provider_score=provider_score,
            )
        )
    return WebSourceSummarizerInputDTO(
        query=_compact_text(query, 1000),
        requested_resource_type=_compact_text(requested_resource_type, 120),
        requested_resource_types=[
            _compact_text(item, 120)
            for item in requested_resource_types
            if str(item or "").strip()
        ],
        output_language=_compact_text(output_language or "same_as_user_query", 80),
        sources=dto_sources,
    )


def _summarizer_source_text(source: dict[str, Any], *, max_chars: int) -> str:
    parts = [
        ("Title", source.get("title")),
        ("Snippet", source.get("snippet")),
        ("Content", source.get("content_preview") or source.get("raw_content")),
    ]
    text = "\n".join(
        f"{label}: {_compact_text(value, max_chars)}"
        for label, value in parts
        if str(value or "").strip()
    )
    return _compact_text(text, max_chars)


def _summarizer_source_context(
    source: dict[str, Any],
    *,
    learning_goal: str,
    requested_resource_type: str,
    requested_resource_types: list[str],
    max_chars: int,
) -> str:
    resource_types = ", ".join(
        _compact_text(item, 80)
        for item in requested_resource_types
        if str(item or "").strip()
    )
    parts = [
        f"Current learning goal: {_compact_text(learning_goal, 240)}",
        f"Requested resource type: {_compact_text(requested_resource_type, 120)}",
        f"Requested resource types: {resource_types}" if resource_types else "",
        f"Search task purpose: {_compact_text(source.get('purpose'), 160)}",
        "Use source_id as the only source identity in output.",
    ]
    return _compact_text("\n".join(part for part in parts if part), max_chars)


def _compact_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


class WebRawSource(BaseModel):
    """Program-normalized search result before fetch/compression.

    Raw provider dictionaries must be converted to this shape before passing
    into Web Research V2 downstream stages.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(...)
    source_id: str = ""
    original_url: str = ""
    canonical_url: str = ""
    title: str = ""
    domain: str = ""
    snippet: str = ""
    raw_content: str = ""
    provider: str = ""
    provider_score: float | None = None
    provider_rank: int = 0
    retrieved_at: str = ""
    subject: str = ""
    role: str = ""
    purpose: str = ""
    search_query: str = ""
    task_priority: float = 0.0
    favicon: str = ""


class WebFetchedSource(WebRawSource):
    """Source after the lightweight fetch/content availability stage."""

    fetch_status: WebFetchStatus = "not_attempted"
    content_chars: int = 0
    content_preview: str = ""
    fetch_error_type: str | None = None
    fetch_error_message_sanitized: str | None = None


class WebCuratedSource(WebFetchedSource):
    """Source after source-level curation.

    Curation is intentionally source-level cleanup/ranking only. Evidence Judge
    V2 remains responsible for final sufficiency decisions.
    """

    curator_keep: bool = True
    curator_reason: str = ""


def validate_web_research_plan(
    parsed: BaseModel,
    *,
    allowed_subjects: list[str],
    max_total_tasks: int,
    max_tasks_per_subject: int,
) -> str:
    """Business validator for Web Research V2 planner output."""

    if not isinstance(parsed, WebResearchPlan):
        return "parsed result is not WebResearchPlan"

    problems: list[str] = []
    tasks = list(parsed.tasks or [])
    allowed = {str(subject).strip() for subject in allowed_subjects if str(subject).strip()}
    max_total_tasks = max(0, int(max_total_tasks or 0))
    max_tasks_per_subject = max(0, int(max_tasks_per_subject or 0))

    if len(tasks) > max_total_tasks:
        problems.append(f"task count {len(tasks)} exceeds max_total_tasks {max_total_tasks}")

    task_ids = [str(task.task_id or "").strip() for task in tasks]
    duplicate_task_ids = sorted([task_id for task_id, count in Counter(task_ids).items() if task_id and count > 1])
    if any(not task_id for task_id in task_ids):
        problems.append("task_id must not be empty")
    if duplicate_task_ids:
        problems.append(f"duplicate task_id values: {duplicate_task_ids}")

    query_keys = [" ".join(str(task.search_query or "").lower().split()) for task in tasks]
    duplicate_queries = sorted([query for query, count in Counter(query_keys).items() if query and count > 1])
    if duplicate_queries:
        problems.append(f"duplicate search_query values: {duplicate_queries}")

    tasks_by_subject: dict[str, int] = defaultdict(int)
    for index, task in enumerate(tasks):
        subject = str(task.subject or "").strip()
        tasks_by_subject[subject] += 1
        if allowed and subject not in allowed:
            problems.append(f"tasks[{index}].subject {subject!r} is not in allowed subjects {sorted(allowed)}")
        if not str(task.search_query or "").strip():
            problems.append(f"tasks[{index}].search_query must not be empty")
        if not str(task.role or "").strip():
            problems.append(f"tasks[{index}].role must not be empty")
        if not str(task.purpose or "").strip():
            problems.append(f"tasks[{index}].purpose must not be empty")
        if not str(task.reason or "").strip():
            problems.append(f"tasks[{index}].reason must not be empty")
        if not 0.0 <= float(task.priority) <= 1.0:
            problems.append(f"tasks[{index}].priority must be between 0 and 1")

    for subject, count in sorted(tasks_by_subject.items()):
        if count > max_tasks_per_subject:
            problems.append(
                f"subject {subject!r} task count {count} exceeds max_tasks_per_subject {max_tasks_per_subject}"
            )

    return "; ".join(problems)


def validate_web_source_summary_batch(
    parsed: BaseModel,
    *,
    expected_source_ids: list[str],
) -> str:
    """Business validator for Web source summarizer output."""

    if not isinstance(parsed, WebSourceSummaryBatch):
        return "parsed result is not WebSourceSummaryBatch"

    problems: list[str] = []
    expected = [str(source_id or "").strip() for source_id in expected_source_ids]
    returned = [str(summary.source_id or "").strip() for summary in parsed.summaries]
    expected_set = set(expected)
    returned_set = set(returned)
    duplicate_ids = sorted([source_id for source_id, count in Counter(returned).items() if source_id and count > 1])
    missing_ids = [source_id for source_id in expected if source_id and source_id not in returned_set]
    unknown_ids = [source_id for source_id in returned if source_id and source_id not in expected_set]

    if any(not source_id for source_id in returned):
        problems.append("source_id must not be empty")
    if missing_ids:
        problems.append(f"missing source_id values: {missing_ids}")
    if duplicate_ids:
        problems.append(f"duplicate source_id values: {duplicate_ids}")
    if unknown_ids:
        problems.append(f"unknown source_id values: {unknown_ids}")
    if len(returned) != len(expected):
        problems.append(f"expected {len(expected)} source summaries, got {len(returned)}")

    for index, summary in enumerate(parsed.summaries):
        if not str(summary.reason or "").strip():
            problems.append(f"summaries[{index}].reason must not be empty")
        if summary.keep and not str(summary.summary or "").strip():
            problems.append(f"summaries[{index}].summary must not be empty when keep=true")
        if summary.keep and not [point for point in summary.coverage_points if str(point or "").strip()]:
            problems.append(f"summaries[{index}].coverage_points must contain at least one item when keep=true")

    return "; ".join(problems)


def domain_from_url(url: str) -> str:
    parsed = urlsplit(str(url or ""))
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def canonicalize_url(url: str) -> str:
    """Normalize URLs enough for dedupe without hiding their origin."""

    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")

    tracking_prefixes = ("utm_",)
    tracking_names = {"fbclid", "gclid", "mc_cid", "mc_eid"}
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if not key.lower().startswith(tracking_prefixes) and key.lower() not in tracking_names
    ]
    query = urlencode(sorted(query_pairs))
    return urlunsplit((scheme, netloc, path, query, ""))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_web_raw_source(
    result: dict[str, Any],
    *,
    task_id: str,
    subject: str,
    role: str,
    purpose: str,
    search_query: str,
    task_priority: float,
    provider: str,
    provider_rank: int,
    retrieved_at: str | None = None,
) -> WebRawSource:
    """Convert a provider result into the WebRawSource contract."""

    original_url = str(result.get("url") or result.get("href") or "")
    raw_content = str(result.get("raw_content") or "")
    snippet = str(result.get("content") or result.get("snippet") or result.get("body") or "")
    score = result.get("score", result.get("tavily_score", result.get("provider_score")))
    try:
        provider_score = float(score) if score is not None else None
    except (TypeError, ValueError):
        provider_score = None
    return WebRawSource(
        task_id=str(task_id or ""),
        original_url=original_url,
        canonical_url=canonicalize_url(original_url),
        title=str(result.get("title") or ""),
        domain=domain_from_url(original_url),
        snippet=snippet,
        raw_content=raw_content,
        provider=str(provider or ""),
        provider_score=provider_score,
        provider_rank=int(provider_rank),
        retrieved_at=str(retrieved_at or utc_now_iso()),
        subject=str(subject or ""),
        role=str(role or ""),
        purpose=str(purpose or ""),
        search_query=str(search_query or ""),
        task_priority=float(task_priority or 0.0),
        favicon=str(result.get("favicon") or ""),
    )


def fetch_source_from_provider_content(
    source: WebRawSource,
    *,
    min_content_chars: int = 1,
    preview_chars: int = 1200,
    sanitize_error,
) -> WebFetchedSource:
    """Use provider-supplied content as the lightweight fetch result."""

    content = str(source.raw_content or source.snippet or "").strip()
    content_chars = len(content)
    if content_chars < max(1, int(min_content_chars or 1)):
        return WebFetchedSource(
            **source.model_dump(),
            fetch_status="failed",
            content_chars=content_chars,
            content_preview=content[:preview_chars],
            fetch_error_type="InsufficientProviderContent",
            fetch_error_message_sanitized=sanitize_error("provider result did not include readable content"),
        )
    return WebFetchedSource(
        **source.model_dump(),
        fetch_status="success",
        content_chars=content_chars,
        content_preview=content[:preview_chars],
    )


def _score_for_dedupe(source: dict[str, Any]) -> tuple[float, float]:
    try:
        score = source.get("tavily_score", source.get("provider_score"))
        tavily_score = float(score if score is not None else -1.0)
    except (TypeError, ValueError):
        tavily_score = -1.0
    try:
        task_priority = float(source.get("task_priority") if source.get("task_priority") is not None else 0.0)
    except (TypeError, ValueError):
        task_priority = 0.0
    return tavily_score, task_priority


def dedupe_sources_by_canonical_url(sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Dedupe sources by canonical_url, keeping higher score then priority."""

    best_by_url: dict[str, dict[str, Any]] = {}
    duplicate_url_count = 0
    missing_url_count = 0

    for source in sources:
        copied = dict(source)
        canonical_url = str(copied.get("canonical_url") or canonicalize_url(str(copied.get("original_url") or "")))
        copied["canonical_url"] = canonical_url
        if not canonical_url:
            missing_url_count += 1
            canonical_url = f"missing-url:{len(best_by_url)}"
            copied["canonical_url"] = ""
        if canonical_url in best_by_url:
            duplicate_url_count += 1
            if _score_for_dedupe(copied) > _score_for_dedupe(best_by_url[canonical_url]):
                best_by_url[canonical_url] = copied
            continue
        best_by_url[canonical_url] = copied

    deduped = list(best_by_url.values())
    deduped.sort(key=lambda source: _score_for_dedupe(source), reverse=True)
    return deduped, {
        "input_count": len(sources),
        "deduped_count": len(deduped),
        "duplicate_url_count": duplicate_url_count,
        "missing_url_count": missing_url_count,
    }
