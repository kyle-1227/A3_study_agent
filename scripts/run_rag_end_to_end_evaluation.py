"""Create and import strict human scores for parent-child RAG answer evaluation.

This command intentionally has no answer-generation or automatic-scoring mode.
It only binds externally produced answer runs to one GoldDataset, exports a
local human review template, and reduces a fully completed template to the
aggregate outcome required by candidate validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TypeVar

from pydantic import BaseModel, ValidationError


PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.rag.parent_child._storage_io import (  # noqa: E402
    model_json_bytes,
    sha256_bytes,
)


_ModelT = TypeVar("_ModelT", bound=BaseModel)
from src.rag.parent_child.end_to_end_evaluation import (  # noqa: E402
    AnswerEvaluationProtocol,
    AnswerRun,
    EndToEndEvaluationError,
    HumanScoreTemplate,
    build_human_score_template,
    outcome_from_completed_template,
    resolve_project_root,
    validate_completed_template,
)
from src.rag.parent_child.evaluation import (  # noqa: E402
    GoldDataset,
    RetrievalEvaluationInput,
)
from src.rag.parent_child.evaluation_gate import EndToEndQualityOutcome  # noqa: E402
from src.rag.parent_child.project_paths import (  # noqa: E402
    ProjectPathError,
    atomic_write_project_bytes,
    require_project_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_shared_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--project-root", type=Path, required=True)
        command.add_argument("--gold-dataset", type=Path, required=True)
        command.add_argument("--baseline-retrieval-input", type=Path, required=True)
        command.add_argument("--candidate-retrieval-input", type=Path, required=True)
        command.add_argument("--baseline-answer-run", type=Path, required=True)
        command.add_argument("--candidate-answer-run", type=Path, required=True)
        command.add_argument("--assessment-protocol", type=Path, required=True)

    export = commands.add_parser("export-template")
    add_shared_inputs(export)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--overwrite", action="store_true")

    imported = commands.add_parser("import-scores")
    add_shared_inputs(imported)
    imported.add_argument("--scored-template", type=Path, required=True)
    imported.add_argument("--output", type=Path, required=True)
    imported.add_argument("--overwrite", action="store_true")
    return parser


def _load_model(path: Path, model_type: type[_ModelT]) -> _ModelT:
    try:
        return model_type.model_validate_json(path.read_bytes())
    except ValidationError as error:
        raise EndToEndEvaluationError(
            "evaluation artifact does not satisfy its strict schema"
        ) from error
    except OSError as error:
        raise EndToEndEvaluationError("unable to read evaluation artifact") from error


def _load_bound_inputs(
    *,
    project_root: Path,
    gold_dataset_path: Path,
    baseline_retrieval_path: Path,
    candidate_retrieval_path: Path,
    baseline_answer_run_path: Path,
    candidate_answer_run_path: Path,
    assessment_protocol_path: Path,
) -> tuple[
    Path,
    GoldDataset,
    str,
    RetrievalEvaluationInput,
    RetrievalEvaluationInput,
    AnswerRun,
    AnswerRun,
    AnswerEvaluationProtocol,
]:
    root = resolve_project_root(project_root)
    gold_path = require_project_file(root, gold_dataset_path)
    baseline_path = require_project_file(root, baseline_retrieval_path)
    candidate_path = require_project_file(root, candidate_retrieval_path)
    baseline_answers_path = require_project_file(root, baseline_answer_run_path)
    candidate_answers_path = require_project_file(root, candidate_answer_run_path)
    protocol_path = require_project_file(root, assessment_protocol_path)
    gold_bytes = gold_path.read_bytes()
    dataset = GoldDataset.model_validate_json(gold_bytes)
    baseline_retrieval = _load_model(baseline_path, RetrievalEvaluationInput)
    candidate_retrieval = _load_model(candidate_path, RetrievalEvaluationInput)
    baseline_answers = _load_model(baseline_answers_path, AnswerRun)
    candidate_answers = _load_model(candidate_answers_path, AnswerRun)
    protocol = _load_model(protocol_path, AnswerEvaluationProtocol)
    return (
        root,
        dataset,
        sha256_bytes(gold_bytes),
        baseline_retrieval,
        candidate_retrieval,
        baseline_answers,
        candidate_answers,
        protocol,
    )


def _write_model(
    *,
    root: Path,
    output_path: Path,
    model: BaseModel,
    overwrite: bool,
) -> Path:
    return atomic_write_project_bytes(
        root,
        output_path,
        model_json_bytes(model),
        overwrite=overwrite,
    )


def run_export_template(
    *,
    project_root: Path,
    gold_dataset_path: Path,
    baseline_retrieval_path: Path,
    candidate_retrieval_path: Path,
    baseline_answer_run_path: Path,
    candidate_answer_run_path: Path,
    assessment_protocol_path: Path,
    output_path: Path,
    overwrite: bool,
) -> HumanScoreTemplate:
    """Write one deterministic, unscored local human-review template."""

    (
        root,
        dataset,
        gold_digest,
        baseline_retrieval,
        candidate_retrieval,
        baseline_answers,
        candidate_answers,
        protocol,
    ) = _load_bound_inputs(
        project_root=project_root,
        gold_dataset_path=gold_dataset_path,
        baseline_retrieval_path=baseline_retrieval_path,
        candidate_retrieval_path=candidate_retrieval_path,
        baseline_answer_run_path=baseline_answer_run_path,
        candidate_answer_run_path=candidate_answer_run_path,
        assessment_protocol_path=assessment_protocol_path,
    )
    template = build_human_score_template(
        dataset=dataset,
        gold_dataset_sha256=gold_digest,
        baseline_retrieval=baseline_retrieval,
        candidate_retrieval=candidate_retrieval,
        baseline_answers=baseline_answers,
        candidate_answers=candidate_answers,
        protocol=protocol,
    )
    _write_model(
        root=root,
        output_path=output_path,
        model=template,
        overwrite=overwrite,
    )
    return template


def run_import_scores(
    *,
    project_root: Path,
    gold_dataset_path: Path,
    baseline_retrieval_path: Path,
    candidate_retrieval_path: Path,
    baseline_answer_run_path: Path,
    candidate_answer_run_path: Path,
    assessment_protocol_path: Path,
    scored_template_path: Path,
    output_path: Path,
    overwrite: bool,
) -> EndToEndQualityOutcome:
    """Revalidate a completed human template and write its aggregate outcome."""

    (
        root,
        dataset,
        gold_digest,
        baseline_retrieval,
        candidate_retrieval,
        baseline_answers,
        candidate_answers,
        protocol,
    ) = _load_bound_inputs(
        project_root=project_root,
        gold_dataset_path=gold_dataset_path,
        baseline_retrieval_path=baseline_retrieval_path,
        candidate_retrieval_path=candidate_retrieval_path,
        baseline_answer_run_path=baseline_answer_run_path,
        candidate_answer_run_path=candidate_answer_run_path,
        assessment_protocol_path=assessment_protocol_path,
    )
    expected = build_human_score_template(
        dataset=dataset,
        gold_dataset_sha256=gold_digest,
        baseline_retrieval=baseline_retrieval,
        candidate_retrieval=candidate_retrieval,
        baseline_answers=baseline_answers,
        candidate_answers=candidate_answers,
        protocol=protocol,
    )
    template_path = require_project_file(root, scored_template_path)
    completed = _load_model(template_path, HumanScoreTemplate)
    validate_completed_template(
        completed_template=completed,
        expected_template=expected,
    )
    outcome = outcome_from_completed_template(completed)
    _write_model(
        root=root,
        output_path=output_path,
        model=outcome,
        overwrite=overwrite,
    )
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "export-template":
            template = run_export_template(
                project_root=args.project_root,
                gold_dataset_path=args.gold_dataset,
                baseline_retrieval_path=args.baseline_retrieval_input,
                candidate_retrieval_path=args.candidate_retrieval_input,
                baseline_answer_run_path=args.baseline_answer_run,
                candidate_answer_run_path=args.candidate_answer_run,
                assessment_protocol_path=args.assessment_protocol,
                output_path=args.output,
                overwrite=args.overwrite,
            )
            print(
                "Human score template written: "
                f"items={len(template.items)}, assessment_source=human"
            )
            return 0
        outcome = run_import_scores(
            project_root=args.project_root,
            gold_dataset_path=args.gold_dataset,
            baseline_retrieval_path=args.baseline_retrieval_input,
            candidate_retrieval_path=args.candidate_retrieval_input,
            baseline_answer_run_path=args.baseline_answer_run,
            candidate_answer_run_path=args.candidate_answer_run,
            assessment_protocol_path=args.assessment_protocol,
            scored_template_path=args.scored_template,
            output_path=args.output,
            overwrite=args.overwrite,
        )
        print(
            "End-to-end quality outcome written: "
            f"scored_query_count={outcome.scored_query_count}, assessment_source=human"
        )
        return 0
    except ValidationError:
        print(
            "End-to-end evaluation failed: invalid strict artifact schema",
            file=sys.stderr,
        )
        return 2
    except (
        EndToEndEvaluationError,
        ProjectPathError,
        FileExistsError,
        OSError,
    ) as error:
        print(
            f"End-to-end evaluation failed: {type(error).__name__}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
