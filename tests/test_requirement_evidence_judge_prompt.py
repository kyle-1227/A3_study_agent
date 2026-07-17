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
    assert "bound_candidates nested inside that same requirement group" in prompt
    assert "no ID was copied from another requirement group" in prompt
    assert "{candidates_json}" not in prompt
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
    assert "against the union of the selected bound candidates" in prompt
    assert "Different selected evidence items may support different clauses" in prompt
    assert "Do not require one candidate to contain an integrated example" in prompt
    assert "unless the acceptance criteria explicitly impose" in prompt
    assert "when any acceptance-criteria clause or operation still lacks" in prompt
    assert (
        "requested resource format is not an extra acceptance-criteria clause" in prompt
    )
    assert "does not mean one cited source must already contain" in prompt
    assert "evaluate those operations separately" in prompt
    assert (
        "related sibling operations that the scoped query intent did not retain"
        in prompt
    )


def test_resource_evidence_planner_prompt_preserves_requested_scope() -> None:
    prompt = load_prompt("resource_evidence_planner", reload=True)

    assert "Scope fidelity rule" in prompt
    assert "may restate only concepts and operations explicitly required" in prompt
    assert "Do not broaden one concept into sibling operations" in prompt
    assert '"list iteration" must not automatically add both' in prompt
    assert "do not invent an integrated-example or single-source condition" in prompt


def test_requirement_evidence_judge_prompt_renders_evidence_limit() -> None:
    prompt = load_prompt("requirement_evidence_judge", reload=True)

    rendered = prompt.format(
        question="How do functions work?",
        learning_goal="Review functions",
        round_index=0,
        requirements_json="[]",
        max_evidence_per_requirement=4,
        attempted_queries_json="[]",
    )

    assert "Each coverage row may select at most 4 evidence_ids." in rendered
