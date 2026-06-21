"""Evidence schemas for dual-source RAG/Web fusion."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    evidence_id: str = Field(
        ...,
        description="Stable id that must exactly match one input EvidenceCandidate.evidence_id.",
    )
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
        "web_article",
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
        "redundant",
        "discard",
    ] = "discard"

    coverage_contribution: str = Field(
        ...,
        description=(
            "Required for every judged item. If keep=true, this must be a non-empty "
            "sentence explaining what exact coverage this evidence contributes to the "
            "user's request. If keep=false, use an empty string. This field describes "
            "what coverage the evidence contributes."
        ),
    )
    reason: str = Field(
        ...,
        description=(
            "Required decision rationale for keeping or rejecting this evidence. This "
            "explains why the grading decision was made."
        ),
    )

    @model_validator(mode="after")
    def validate_keep_requires_coverage_contribution(self):
        if self.keep and not self.coverage_contribution.strip():
            raise ValueError(
                "coverage_contribution must not be empty when keep=true "
                f"for evidence_id={self.evidence_id}"
            )
        return self


class EvidenceCoverageGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = ""
    role: str = ""
    gap: str = Field(
        ...,
        description="A concise description of important coverage that is missing or weak.",
    )
    suggested_search_query: str = Field(
        ...,
        description="Concise English-first Tavily search query to retrieve evidence for this gap.",
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


class EvidenceGradeBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judged_evidence: list[EvidenceJudgeItem] = Field(default_factory=list, max_length=8)


class EvidenceSufficiencyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_evidence_state: Literal[
        "sufficient",
        "partially_sufficient",
        "insufficient",
    ] = "insufficient"
    answerability: Literal[
        "can_answer",
        "can_answer_with_caveats",
        "cannot_answer",
    ] = "cannot_answer"
    need_more_local_rag: bool = False
    need_more_web_research: bool = False
    coverage_gaps: list[EvidenceCoverageGap] = Field(default_factory=list, max_length=10)
    decision_summary: str = Field("", max_length=600)


class EvidenceJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_evidence_state: Literal[
        "sufficient",
        "partially_sufficient",
        "insufficient",
    ] = "insufficient"

    need_more_web_research: bool = False
    judged_evidence: list[EvidenceJudgeItem] = Field(default_factory=list)
    coverage_gaps: list[EvidenceCoverageGap] = Field(default_factory=list)
    decision_summary: str = ""
