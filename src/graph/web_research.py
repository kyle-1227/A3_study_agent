"""Pure Web Research V2 schemas, validators, and URL helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
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


class WebResearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(...)
    subject: str = Field(...)
    role: str = Field(...)
    purpose: str = Field(...)
    search_query: str = Field(...)
    reason: str = Field(...)
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


def _score_for_dedupe(source: dict[str, Any]) -> tuple[float, float]:
    try:
        tavily_score = float(source.get("tavily_score") if source.get("tavily_score") is not None else -1.0)
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
