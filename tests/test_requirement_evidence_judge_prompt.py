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
