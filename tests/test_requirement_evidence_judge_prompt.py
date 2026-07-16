"""Contract checks for the requirement evidence judge prompt."""

from src.config import load_prompt


def test_requirement_evidence_judge_prompt_has_exact_gap_query_matrix() -> None:
    prompt = load_prompt("requirement_evidence_judge", reload=True)

    assert "partial or missing with local_only" in prompt
    assert "partial or missing with web_only" in prompt
    assert "partial or missing with local_and_web" in prompt
    assert "partial or missing with local_then_web_on_gap" in prompt
    assert "Never populate both fields for local_then_web_on_gap" in prompt
    assert "Determine the local_then_web_on_gap stage only" in prompt
    assert "same requirement_id and source_type" in prompt
    assert "explicit query text in attempted_queries_json" in prompt
    assert "whitespace-only or punctuation-only changes" in prompt
    assert "local_and_web rule is never staged" in prompt
    assert "attempted query history must not suppress either field" in prompt
    assert "Before returning, self-check every row against both binding" in prompt
    assert "eligible_evidence_ids in each requirement" in prompt
    assert "never copy an evidence_id between requirements" in prompt
    assert (
        "every selected evidence_id belongs to that row's eligible_evidence_ids"
        in prompt
    )
    assert (
        "Copy round_index into every row; never omit it during a correction" in prompt
    )
    assert "required_incomplete_query_shape attached to every requirement" in prompt
    assert "This requirement applies again in every supplement round" in prompt


def test_requirement_evidence_judge_prompt_renders_evidence_limit() -> None:
    prompt = load_prompt("requirement_evidence_judge", reload=True)

    rendered = prompt.format(
        question="How do functions work?",
        learning_goal="Review functions",
        round_index=0,
        requirements_json="[]",
        candidates_json="[]",
        max_evidence_per_requirement=4,
        attempted_queries_json="[]",
    )

    assert "Each coverage row may select at most 4 evidence_ids." in rendered
