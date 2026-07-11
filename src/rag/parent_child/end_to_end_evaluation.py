"""Strict offline contracts for human end-to-end RAG evaluation.

The module deliberately does not generate answers or call an evaluation model.
It binds externally produced answer runs to one fixed GoldDataset and emits a
human-review template.  Only a fully completed, revalidated template can be
reduced to the aggregate outcome consumed by the candidate validator.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.rag.parent_child._storage_io import canonical_json_bytes, sha256_bytes
from src.rag.parent_child.evaluation import (
    GoldDataset,
    GoldEvidenceSpan,
    GoldQuery,
    RetrievalEvaluationInput,
)
from src.rag.parent_child.evaluation_gate import EndToEndQualityOutcome
from src.rag.parent_child.project_paths import (
    ProjectPathError,
    resolve_project_root,
)


_SHA256_RE = "^[0-9a-f]{64}$"
_DOC_ID_RE = "^doc_[0-9a-f]{40}$"


class EndToEndEvaluationError(ValueError):
    """Raised when answer, score, or retrieval evaluation artifacts disagree."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and already stripped")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise ValueError(f"{field_name} must not contain path separators or NUL")
    return value


def _validate_subject(value: str) -> str:
    _validate_identifier(value, field_name="subject")
    if value != value.casefold():
        raise ValueError("subject must already be case-folded")
    if (
        value.startswith("_")
        or value.endswith("_")
        or "__" in value
        or not all(character.isalnum() or character == "_" for character in value)
    ):
        raise ValueError("subject must be a normalized identifier")
    return value


def _validate_source_relpath(value: str) -> str:
    if not value or value != value.strip() or "\\" in value or "\x00" in value:
        raise ValueError(
            "source_relpath must be non-empty, stripped, POSIX, and contain no NUL"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("source_relpath must be a contained POSIX relative path")
    return path.as_posix()


def _validate_text(value: str, *, field_name: str) -> str:
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL")
    return value


class AnswerCitation(_StrictFrozenModel):
    """A policy-independent citation coordinate shown to the human reviewer."""

    schema_version: Literal["answer_citation_v1"]
    citation_id: str
    source_relpath: str
    doc_id: str = Field(pattern=_DOC_ID_RE)
    pagination_kind: Literal["physical", "logical"]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    section_path: tuple[str, ...] | None

    @field_validator("citation_id")
    @classmethod
    def _citation_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="citation_id")

    @field_validator("source_relpath")
    @classmethod
    def _source_relpath(cls, value: str) -> str:
        return _validate_source_relpath(value)

    @field_validator("section_path")
    @classmethod
    def _section_path(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("section_path must be None or a non-empty tuple")
        for item in value:
            if not item or item != item.strip() or "\x00" in item:
                raise ValueError("section_path items must be non-empty and stripped")
        return value

    @model_validator(mode="after")
    def _coordinate_range(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        if self.end_char <= self.start_char:
            raise ValueError("citation cleaned-character span must be non-empty")
        return self


class AnswerRunItem(_StrictFrozenModel):
    """One externally produced answer and the context consumed to produce it."""

    schema_version: Literal["answer_run_item_v1"]
    query_id: str
    answer: str
    citations: tuple[AnswerCitation, ...]
    context_tokens: int = Field(ge=0)

    @field_validator("query_id")
    @classmethod
    def _query_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="query_id")

    @field_validator("answer")
    @classmethod
    def _answer(cls, value: str) -> str:
        # An empty answer is still an explicit production result that a human can
        # score as incorrect; a missing answer is rejected by the schema instead.
        return _validate_text(value, field_name="answer")

    @model_validator(mode="after")
    def _unique_citation_ids(self) -> Self:
        citation_ids = tuple(citation.citation_id for citation in self.citations)
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("citations must not contain duplicate citation_id values")
        return self


class AnswerRun(_StrictFrozenModel):
    """Complete answer run tied to exactly one retrieval run and GoldDataset."""

    schema_version: Literal["answer_run_v1"]
    answer_run_id: str
    dataset_id: str
    gold_dataset_sha256: str = Field(pattern=_SHA256_RE)
    retrieval_run_id: str
    answer_model_fingerprint: str = Field(pattern=_SHA256_RE)
    answers: tuple[AnswerRunItem, ...] = Field(min_length=1)

    @field_validator("answer_run_id", "dataset_id", "retrieval_run_id")
    @classmethod
    def _identifiers(cls, value: str, info: object) -> str:
        return _validate_identifier(
            value, field_name=str(getattr(info, "field_name", "identifier"))
        )

    @model_validator(mode="after")
    def _ordered_unique_answers(self) -> Self:
        query_ids = tuple(answer.query_id for answer in self.answers)
        if query_ids != tuple(sorted(query_ids)):
            raise ValueError("answers must be sorted by query_id")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("answers must not contain duplicate query_id values")
        return self


class AnswerEvaluationProtocol(_StrictFrozenModel):
    """Explicit, versioned human assessment protocol for an answer comparison."""

    schema_version: Literal["answer_evaluation_protocol_v1"]
    protocol_id: str
    answer_model_fingerprint: str = Field(pattern=_SHA256_RE)
    rubric_version: str
    reviewer_instructions: str

    @field_validator("protocol_id", "rubric_version")
    @classmethod
    def _identifiers(cls, value: str, info: object) -> str:
        return _validate_identifier(
            value, field_name=str(getattr(info, "field_name", "identifier"))
        )

    @field_validator("reviewer_instructions")
    @classmethod
    def _reviewer_instructions(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("reviewer_instructions must be non-empty and stripped")
        return _validate_text(value, field_name="reviewer_instructions")


class ArmHumanScore(_StrictFrozenModel):
    """Human judgement for one answer arm; no nullable or default pass values."""

    schema_version: Literal["arm_human_score_v1"]
    answer_correct: bool
    citations_supported: bool
    hallucination_present: bool


class HumanScoreTemplateItem(_StrictFrozenModel):
    """Local content-bearing review item; scores are completed after export."""

    schema_version: Literal["human_score_template_item_v1"]
    query_id: str
    subject: str
    query: str
    gold_spans: tuple[GoldEvidenceSpan, ...] = Field(min_length=1)
    baseline_answer: str
    baseline_citations: tuple[AnswerCitation, ...]
    baseline_context_tokens: int = Field(ge=0)
    candidate_answer: str
    candidate_citations: tuple[AnswerCitation, ...]
    candidate_context_tokens: int = Field(ge=0)
    baseline_score: ArmHumanScore | None
    candidate_score: ArmHumanScore | None

    @field_validator("query_id")
    @classmethod
    def _query_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="query_id")

    @field_validator("subject")
    @classmethod
    def _subject(cls, value: str) -> str:
        return _validate_subject(value)

    @field_validator("query", "baseline_answer", "candidate_answer")
    @classmethod
    def _text(cls, value: str, info: object) -> str:
        return _validate_text(
            value, field_name=str(getattr(info, "field_name", "text"))
        )


class HumanScoreTemplate(_StrictFrozenModel):
    """Auditable local template whose immutable content is revalidated on import."""

    schema_version: Literal["human_score_template_v1"]
    dataset_id: str
    gold_dataset_sha256: str = Field(pattern=_SHA256_RE)
    baseline_run_id: str
    candidate_run_id: str
    baseline_answer_run_id: str
    candidate_answer_run_id: str
    answer_model_fingerprint: str = Field(pattern=_SHA256_RE)
    assessment_protocol_sha256: str = Field(pattern=_SHA256_RE)
    assessment_source: Literal["human"]
    items: tuple[HumanScoreTemplateItem, ...] = Field(min_length=1)

    @field_validator(
        "dataset_id",
        "baseline_run_id",
        "candidate_run_id",
        "baseline_answer_run_id",
        "candidate_answer_run_id",
    )
    @classmethod
    def _identifiers(cls, value: str, info: object) -> str:
        return _validate_identifier(
            value, field_name=str(getattr(info, "field_name", "identifier"))
        )

    @model_validator(mode="after")
    def _ordered_unique_items(self) -> Self:
        query_ids = tuple(item.query_id for item in self.items)
        if query_ids != tuple(sorted(query_ids)):
            raise ValueError("items must be sorted by query_id")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("items must not contain duplicate query_id values")
        if self.baseline_run_id == self.candidate_run_id:
            raise ValueError("baseline_run_id and candidate_run_id must differ")
        if self.baseline_answer_run_id == self.candidate_answer_run_id:
            raise ValueError(
                "baseline_answer_run_id and candidate_answer_run_id must differ"
            )
        return self


def assessment_protocol_sha256(protocol: AnswerEvaluationProtocol) -> str:
    """Return the canonical digest recorded in all dependent artifacts."""

    return sha256_bytes(canonical_json_bytes(protocol.model_dump(mode="json")))


def _assert_dataset_query_set(
    *,
    dataset: GoldDataset,
    query_ids: tuple[str, ...],
    artifact_name: str,
) -> None:
    expected = {query.query_id for query in dataset.queries}
    actual = set(query_ids)
    if actual != expected or len(query_ids) != len(actual):
        raise EndToEndEvaluationError(
            f"{artifact_name} query set must exactly match the GoldDataset"
        )


def _validate_retrieval_input(
    *,
    dataset: GoldDataset,
    gold_dataset_sha256: str,
    retrieval_input: RetrievalEvaluationInput,
    expected_parent_aware: bool,
    arm: Literal["baseline", "candidate"],
) -> None:
    if retrieval_input.dataset_id != dataset.dataset_id:
        raise EndToEndEvaluationError(
            f"{arm} retrieval dataset_id does not match GoldDataset"
        )
    if retrieval_input.gold_dataset_sha256 != gold_dataset_sha256:
        raise EndToEndEvaluationError(
            f"{arm} retrieval GoldDataset digest does not match the input file"
        )
    if retrieval_input.parent_aware is not expected_parent_aware:
        raise EndToEndEvaluationError(
            f"{arm} retrieval parent_aware contract is invalid"
        )
    expected_implementation = (
        "flat_baseline" if arm == "baseline" else "parent_child_candidate"
    )
    if retrieval_input.implementation_kind != expected_implementation:
        raise EndToEndEvaluationError(
            f"{arm} retrieval implementation identity is invalid"
        )
    _assert_dataset_query_set(
        dataset=dataset,
        query_ids=tuple(result.query_id for result in retrieval_input.results),
        artifact_name=f"{arm} retrieval input",
    )


def _validate_answer_run(
    *,
    dataset: GoldDataset,
    gold_dataset_sha256: str,
    answer_run: AnswerRun,
    retrieval_input: RetrievalEvaluationInput,
    protocol: AnswerEvaluationProtocol,
    arm: Literal["baseline", "candidate"],
) -> None:
    if answer_run.dataset_id != dataset.dataset_id:
        raise EndToEndEvaluationError(f"{arm} answer run dataset_id does not match")
    if answer_run.gold_dataset_sha256 != gold_dataset_sha256:
        raise EndToEndEvaluationError(
            f"{arm} answer run GoldDataset digest does not match the input file"
        )
    if answer_run.retrieval_run_id != retrieval_input.run_id:
        raise EndToEndEvaluationError(
            f"{arm} answer run is not bound to the supplied retrieval run"
        )
    if answer_run.answer_model_fingerprint != protocol.answer_model_fingerprint:
        raise EndToEndEvaluationError(
            f"{arm} answer model fingerprint does not match the assessment protocol"
        )
    _assert_dataset_query_set(
        dataset=dataset,
        query_ids=tuple(answer.query_id for answer in answer_run.answers),
        artifact_name=f"{arm} answer run",
    )


def _rollout_eligible_human_queries(dataset: GoldDataset) -> tuple[GoldQuery, ...]:
    queries = tuple(
        query
        for query in dataset.queries
        if query.eligible_for_rollout
        and query.dataset_kind in {"human_gold", "historical_annotated"}
    )
    if not queries:
        raise EndToEndEvaluationError(
            "no rollout-eligible human or historical GoldDataset queries exist"
        )
    return tuple(sorted(queries, key=lambda query: query.query_id))


def build_human_score_template(
    *,
    dataset: GoldDataset,
    gold_dataset_sha256: str,
    baseline_retrieval: RetrievalEvaluationInput,
    candidate_retrieval: RetrievalEvaluationInput,
    baseline_answers: AnswerRun,
    candidate_answers: AnswerRun,
    protocol: AnswerEvaluationProtocol,
) -> HumanScoreTemplate:
    """Build the only permitted draft score template for one fixed comparison."""

    _validate_retrieval_input(
        dataset=dataset,
        gold_dataset_sha256=gold_dataset_sha256,
        retrieval_input=baseline_retrieval,
        expected_parent_aware=False,
        arm="baseline",
    )
    _validate_retrieval_input(
        dataset=dataset,
        gold_dataset_sha256=gold_dataset_sha256,
        retrieval_input=candidate_retrieval,
        expected_parent_aware=True,
        arm="candidate",
    )
    if baseline_retrieval.run_id == candidate_retrieval.run_id:
        raise EndToEndEvaluationError(
            "baseline and candidate retrieval run IDs must differ"
        )
    if (
        baseline_retrieval.embedding_fingerprint
        != candidate_retrieval.embedding_fingerprint
    ):
        raise EndToEndEvaluationError(
            "baseline and candidate retrieval embedding fingerprints differ"
        )
    _validate_answer_run(
        dataset=dataset,
        gold_dataset_sha256=gold_dataset_sha256,
        answer_run=baseline_answers,
        retrieval_input=baseline_retrieval,
        protocol=protocol,
        arm="baseline",
    )
    _validate_answer_run(
        dataset=dataset,
        gold_dataset_sha256=gold_dataset_sha256,
        answer_run=candidate_answers,
        retrieval_input=candidate_retrieval,
        protocol=protocol,
        arm="candidate",
    )
    if baseline_answers.answer_run_id == candidate_answers.answer_run_id:
        raise EndToEndEvaluationError(
            "baseline and candidate answer run IDs must differ"
        )

    baseline_by_query = {answer.query_id: answer for answer in baseline_answers.answers}
    candidate_by_query = {
        answer.query_id: answer for answer in candidate_answers.answers
    }
    items = tuple(
        HumanScoreTemplateItem(
            schema_version="human_score_template_item_v1",
            query_id=query.query_id,
            subject=query.subject,
            query=query.query,
            gold_spans=query.gold_spans,
            baseline_answer=baseline_by_query[query.query_id].answer,
            baseline_citations=baseline_by_query[query.query_id].citations,
            baseline_context_tokens=baseline_by_query[query.query_id].context_tokens,
            candidate_answer=candidate_by_query[query.query_id].answer,
            candidate_citations=candidate_by_query[query.query_id].citations,
            candidate_context_tokens=candidate_by_query[query.query_id].context_tokens,
            baseline_score=None,
            candidate_score=None,
        )
        for query in _rollout_eligible_human_queries(dataset)
    )
    return HumanScoreTemplate(
        schema_version="human_score_template_v1",
        dataset_id=dataset.dataset_id,
        gold_dataset_sha256=gold_dataset_sha256,
        baseline_run_id=baseline_retrieval.run_id,
        candidate_run_id=candidate_retrieval.run_id,
        baseline_answer_run_id=baseline_answers.answer_run_id,
        candidate_answer_run_id=candidate_answers.answer_run_id,
        answer_model_fingerprint=protocol.answer_model_fingerprint,
        assessment_protocol_sha256=assessment_protocol_sha256(protocol),
        assessment_source="human",
        items=items,
    )


def _without_scores(template: HumanScoreTemplate) -> HumanScoreTemplate:
    return template.model_copy(
        update={
            "items": tuple(
                item.model_copy(
                    update={"baseline_score": None, "candidate_score": None}
                )
                for item in template.items
            )
        }
    )


def validate_completed_template(
    *,
    completed_template: HumanScoreTemplate,
    expected_template: HumanScoreTemplate,
) -> None:
    """Reject a scored template whose review material was changed after export."""

    if _without_scores(completed_template) != expected_template:
        raise EndToEndEvaluationError(
            "scored template content or bindings differ from the exported template"
        )
    incomplete = tuple(
        item.query_id
        for item in completed_template.items
        if item.baseline_score is None or item.candidate_score is None
    )
    if incomplete:
        raise EndToEndEvaluationError("human scores are incomplete")


def outcome_from_completed_template(
    completed_template: HumanScoreTemplate,
) -> EndToEndQualityOutcome:
    """Aggregate an already validated, fully scored human review template."""

    if completed_template.assessment_source != "human":
        raise EndToEndEvaluationError("only human assessment can create this outcome")
    if any(
        item.baseline_score is None or item.candidate_score is None
        for item in completed_template.items
    ):
        raise EndToEndEvaluationError("human scores are incomplete")

    count = len(completed_template.items)
    baseline_scores = tuple(item.baseline_score for item in completed_template.items)
    candidate_scores = tuple(item.candidate_score for item in completed_template.items)
    if any(score is None for score in baseline_scores + candidate_scores):
        raise EndToEndEvaluationError("human scores are incomplete")
    # The prior guard narrows these tuples at runtime without substituting values.
    baseline = tuple(score for score in baseline_scores if score is not None)
    candidate = tuple(score for score in candidate_scores if score is not None)
    baseline_tokens = sum(
        item.baseline_context_tokens for item in completed_template.items
    )
    candidate_tokens = sum(
        item.candidate_context_tokens for item in completed_template.items
    )
    return EndToEndQualityOutcome(
        schema_version="end_to_end_quality_outcome_v2",
        dataset_id=completed_template.dataset_id,
        gold_dataset_sha256=completed_template.gold_dataset_sha256,
        baseline_run_id=completed_template.baseline_run_id,
        candidate_run_id=completed_template.candidate_run_id,
        answer_model_fingerprint=completed_template.answer_model_fingerprint,
        assessment_protocol_sha256=completed_template.assessment_protocol_sha256,
        assessment_source="human",
        scored_query_count=count,
        baseline_answer_correctness=sum(score.answer_correct for score in baseline)
        / count,
        candidate_answer_correctness=sum(score.answer_correct for score in candidate)
        / count,
        baseline_citation_support=sum(score.citations_supported for score in baseline)
        / count,
        candidate_citation_support=sum(score.citations_supported for score in candidate)
        / count,
        baseline_hallucination_rate=sum(
            score.hallucination_present for score in baseline
        )
        / count,
        candidate_hallucination_rate=sum(
            score.hallucination_present for score in candidate
        )
        / count,
        baseline_context_tokens_total=baseline_tokens,
        candidate_context_tokens_total=candidate_tokens,
        baseline_context_tokens_mean=baseline_tokens / count,
        candidate_context_tokens_mean=candidate_tokens / count,
    )


__all__ = [
    "AnswerCitation",
    "AnswerEvaluationProtocol",
    "AnswerRun",
    "AnswerRunItem",
    "ArmHumanScore",
    "EndToEndEvaluationError",
    "ProjectPathError",
    "HumanScoreTemplate",
    "HumanScoreTemplateItem",
    "assessment_protocol_sha256",
    "build_human_score_template",
    "outcome_from_completed_template",
    "resolve_project_root",
    "validate_completed_template",
]
