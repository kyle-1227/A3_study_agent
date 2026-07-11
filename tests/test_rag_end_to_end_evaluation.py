from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.run_rag_end_to_end_evaluation import (
    _parser,
    run_export_template,
    run_import_scores,
)
from src.rag.parent_child._storage_io import model_json_bytes, sha256_bytes
from src.rag.parent_child.end_to_end_evaluation import (
    AnswerEvaluationProtocol,
    AnswerRun,
    AnswerRunItem,
    ArmHumanScore,
    EndToEndEvaluationError,
    HumanScoreTemplate,
    build_human_score_template,
    outcome_from_completed_template,
    validate_completed_template,
)
from src.rag.parent_child.evaluation import (
    GoldDataset,
    GoldEvidenceSpan,
    GoldQuery,
    QueryRetrievalResult,
    RetrievalEvaluationInput,
)
from src.rag.parent_child.evaluation_gate import EndToEndQualityOutcome
from src.rag.parent_child.project_paths import ProjectPathError, require_project_file


DOC_ID = "doc_" + "a" * 40
EMBEDDING_FINGERPRINT = "b" * 64
MODEL_FINGERPRINT = "c" * 64


def _dataset() -> GoldDataset:
    span = GoldEvidenceSpan(
        schema_version="gold_evidence_span_v1",
        gold_span_id="gold_human-question",
        source_group_id="math-book-a",
        source_relpath="math/book-a.md",
        doc_id=DOC_ID,
        pagination_kind="logical",
        page_start=1,
        page_end=1,
        start_char=0,
        end_char=12,
        section_path=("Limits",),
        relevance_grade=3,
    )
    return GoldDataset(
        schema_version="gold_dataset_v1",
        dataset_id="gold-v1",
        queries=(
            GoldQuery(
                schema_version="gold_query_v1",
                query_id="human-question",
                subject="math",
                query="What is a limit?",
                dataset_kind="human_gold",
                eligible_for_rollout=True,
                gold_spans=(span,),
            ),
            GoldQuery(
                schema_version="gold_query_v1",
                query_id="smoke-question",
                subject="math",
                query="Synthetic smoke question",
                dataset_kind="synthetic_smoke",
                eligible_for_rollout=False,
                gold_spans=(span,),
            ),
        ),
    )


def _retrieval(
    dataset: GoldDataset,
    digest: str,
    *,
    arm: str,
) -> RetrievalEvaluationInput:
    is_baseline = arm == "baseline"
    return RetrievalEvaluationInput(
        schema_version="retrieval_evaluation_input_v2",
        run_id="flat-run" if is_baseline else "candidate-run",
        dataset_id=dataset.dataset_id,
        gold_dataset_sha256=digest,
        embedding_fingerprint=EMBEDDING_FINGERPRINT,
        retrieval_fingerprint=("d" if is_baseline else "e") * 64,
        implementation_kind=(
            "flat_baseline" if is_baseline else "parent_child_candidate"
        ),
        artifact_manifest_sha256=("f" if is_baseline else "1") * 64,
        generation_id=None if is_baseline else "candidate-generation-001",
        parent_aware=not is_baseline,
        results=tuple(
            QueryRetrievalResult(
                schema_version="query_retrieval_result_v1",
                query_id=query.query_id,
                subject=query.subject,
                hits=(),
            )
            for query in dataset.queries
        ),
    )


def _answer_run(
    dataset: GoldDataset,
    digest: str,
    retrieval: RetrievalEvaluationInput,
    *,
    answer_run_id: str,
    answer_prefix: str,
    context_tokens: int,
) -> AnswerRun:
    return AnswerRun(
        schema_version="answer_run_v1",
        answer_run_id=answer_run_id,
        dataset_id=dataset.dataset_id,
        gold_dataset_sha256=digest,
        retrieval_run_id=retrieval.run_id,
        answer_model_fingerprint=MODEL_FINGERPRINT,
        answers=tuple(
            AnswerRunItem(
                schema_version="answer_run_item_v1",
                query_id=query.query_id,
                answer=f"{answer_prefix}: {query.query}",
                citations=(),
                context_tokens=context_tokens,
            )
            for query in dataset.queries
        ),
    )


def _protocol() -> AnswerEvaluationProtocol:
    return AnswerEvaluationProtocol(
        schema_version="answer_evaluation_protocol_v1",
        protocol_id="manual-v1",
        answer_model_fingerprint=MODEL_FINGERPRINT,
        rubric_version="rubric-v1",
        reviewer_instructions="Score each answer against the displayed evidence.",
    )


def _complete(template: HumanScoreTemplate) -> HumanScoreTemplate:
    return template.model_copy(
        update={
            "items": tuple(
                item.model_copy(
                    update={
                        "baseline_score": ArmHumanScore(
                            schema_version="arm_human_score_v1",
                            answer_correct=True,
                            citations_supported=True,
                            hallucination_present=False,
                        ),
                        "candidate_score": ArmHumanScore(
                            schema_version="arm_human_score_v1",
                            answer_correct=False,
                            citations_supported=False,
                            hallucination_present=True,
                        ),
                    }
                )
                for item in template.items
            )
        }
    )


def _template() -> HumanScoreTemplate:
    dataset = _dataset()
    digest = "a" * 64
    baseline = _retrieval(dataset, digest, arm="baseline")
    candidate = _retrieval(dataset, digest, arm="candidate")
    return build_human_score_template(
        dataset=dataset,
        gold_dataset_sha256=digest,
        baseline_retrieval=baseline,
        candidate_retrieval=candidate,
        baseline_answers=_answer_run(
            dataset,
            digest,
            baseline,
            answer_run_id="baseline-answer-run",
            answer_prefix="baseline",
            context_tokens=100,
        ),
        candidate_answers=_answer_run(
            dataset,
            digest,
            candidate,
            answer_run_id="candidate-answer-run",
            answer_prefix="candidate",
            context_tokens=120,
        ),
        protocol=_protocol(),
    )


def test_human_template_excludes_synthetic_and_aggregates_only_completed_scores() -> (
    None
):
    template = _template()

    assert tuple(item.query_id for item in template.items) == ("human-question",)
    assert template.items[0].baseline_score is None
    assert template.assessment_source == "human"

    completed = _complete(template)
    validate_completed_template(
        completed_template=completed,
        expected_template=template,
    )
    outcome = outcome_from_completed_template(completed)

    assert outcome.schema_version == "end_to_end_quality_outcome_v2"
    assert outcome.scored_query_count == 1
    assert outcome.baseline_answer_correctness == 1.0
    assert outcome.candidate_answer_correctness == 0.0
    assert outcome.baseline_context_tokens_total == 100
    assert outcome.candidate_context_tokens_mean == 120.0


def test_incomplete_or_modified_human_template_fails_closed() -> None:
    template = _template()
    with pytest.raises(EndToEndEvaluationError, match="incomplete"):
        outcome_from_completed_template(template)

    completed = _complete(template)
    changed = completed.model_copy(
        update={
            "items": (
                completed.items[0].model_copy(update={"candidate_answer": "changed"}),
            )
        }
    )
    with pytest.raises(EndToEndEvaluationError, match="content or bindings"):
        validate_completed_template(
            completed_template=changed,
            expected_template=template,
        )


def test_outcome_requires_explicit_human_bindings_and_consistent_token_means() -> None:
    payload = outcome_from_completed_template(_complete(_template())).model_dump(
        mode="python"
    )
    payload["assessment_source"] = "automatic"
    with pytest.raises(ValidationError):
        EndToEndQualityOutcome.model_validate(payload)

    payload = outcome_from_completed_template(_complete(_template())).model_dump(
        mode="python"
    )
    payload["candidate_context_tokens_mean"] = 0.0
    with pytest.raises(ValidationError, match="candidate_context_tokens_mean"):
        EndToEndQualityOutcome.model_validate(payload)


def _write_model(project: Path, relative_path: str, model: object) -> Path:
    path = project / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "model_dump"):
        raise TypeError("test artifact must be a Pydantic model")
    path.write_bytes(model_json_bytes(model))
    return path


def test_cli_export_and_import_bind_inputs_and_keep_answer_text_out_of_outcome(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    dataset = _dataset()
    gold_path = _write_model(project, "data/evaluation/gold.json", dataset)
    digest = sha256_bytes(gold_path.read_bytes())
    baseline = _retrieval(dataset, digest, arm="baseline")
    candidate = _retrieval(dataset, digest, arm="candidate")
    baseline_answers = _answer_run(
        dataset,
        digest,
        baseline,
        answer_run_id="baseline-answer-run",
        answer_prefix="TEST_SECRET_SENTINEL",
        context_tokens=100,
    )
    candidate_answers = _answer_run(
        dataset,
        digest,
        candidate,
        answer_run_id="candidate-answer-run",
        answer_prefix="candidate",
        context_tokens=120,
    )
    _write_model(project, "artifacts/baseline-retrieval.json", baseline)
    _write_model(project, "artifacts/candidate-retrieval.json", candidate)
    _write_model(project, "artifacts/baseline-answers.json", baseline_answers)
    _write_model(project, "artifacts/candidate-answers.json", candidate_answers)
    _write_model(project, "artifacts/protocol.json", _protocol())

    template = run_export_template(
        project_root=project,
        gold_dataset_path=Path("data/evaluation/gold.json"),
        baseline_retrieval_path=Path("artifacts/baseline-retrieval.json"),
        candidate_retrieval_path=Path("artifacts/candidate-retrieval.json"),
        baseline_answer_run_path=Path("artifacts/baseline-answers.json"),
        candidate_answer_run_path=Path("artifacts/candidate-answers.json"),
        assessment_protocol_path=Path("artifacts/protocol.json"),
        output_path=Path("artifacts/template.json"),
        overwrite=False,
    )
    _write_model(project, "artifacts/scored-template.json", _complete(template))

    outcome = run_import_scores(
        project_root=project,
        gold_dataset_path=Path("data/evaluation/gold.json"),
        baseline_retrieval_path=Path("artifacts/baseline-retrieval.json"),
        candidate_retrieval_path=Path("artifacts/candidate-retrieval.json"),
        baseline_answer_run_path=Path("artifacts/baseline-answers.json"),
        candidate_answer_run_path=Path("artifacts/candidate-answers.json"),
        assessment_protocol_path=Path("artifacts/protocol.json"),
        scored_template_path=Path("artifacts/scored-template.json"),
        output_path=Path("artifacts/outcome.json"),
        overwrite=False,
    )

    assert outcome.assessment_source == "human"
    outcome_bytes = (project / "artifacts/outcome.json").read_bytes()
    assert b"TEST_SECRET_SENTINEL" not in outcome_bytes
    assert b"What is a limit?" not in outcome_bytes


def test_cli_requires_explicit_subcommand_arguments_and_rejects_outside_or_linked_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args([])

    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ProjectPathError):
        require_project_file(project, outside)

    linked = project / "linked.json"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("local Windows policy does not permit symlink creation")
    with pytest.raises(ProjectPathError):
        require_project_file(project, linked)
