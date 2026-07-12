from __future__ import annotations

from pathlib import Path

import pytest

from src.graph.resource_validation import (
    RESOURCE_VALIDATORS,
    validate_renderable_resource_result,
)


@pytest.fixture
def artifact_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    roots = {
        "mindmap": tmp_path / "mindmaps",
        "quiz": tmp_path / "exercises",
        "review_doc": tmp_path / "review-docs",
        "code_practice": tmp_path / "code-practice",
        "video_script": tmp_path / "video-scripts",
        "video_animation": tmp_path / "video-animations",
    }
    env_names = {
        "mindmap": "MINDMAP_ARTIFACT_DIR",
        "quiz": "EXERCISE_ARTIFACT_DIR",
        "review_doc": "REVIEW_DOC_ARTIFACT_DIR",
        "code_practice": "CODE_PRACTICE_ARTIFACT_DIR",
        "video_script": "VIDEO_SCRIPT_ARTIFACT_DIR",
        "video_animation": "VIDEO_ANIMATION_ARTIFACT_DIR",
    }
    for key, root in roots.items():
        root.mkdir(parents=True)
        monkeypatch.setenv(env_names[key], str(root))
    return roots


def _local_artifact(
    root: Path,
    *,
    artifact_id: str,
    filename: str,
    url_field: str,
    filename_field: str,
    body: bytes = b"renderable",
) -> dict[str, str]:
    target = root / artifact_id / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    return {
        "artifact_id": artifact_id,
        filename_field: filename,
        url_field: f"/artifacts/{artifact_id}/{filename}",
    }


def _study_plan() -> dict:
    phase = {
        "title": "Foundation",
        "duration": "2 weeks",
        "goals": ["Understand concepts"],
        "tasks": ["Read and practice"],
        "resources": ["Judged source"],
        "practice": ["Solve exercises"],
        "checkpoints": ["Pass review"],
    }
    return {
        "title": "Machine learning plan",
        "learner_profile_summary": "Beginner learner",
        "overall_goal": "Build a reliable foundation",
        "phases": [phase, {**phase, "title": "Application"}],
        "weekly_schedule": ["Weekdays: 60 minutes"],
        "milestones": ["Complete first project"],
        "practice_tasks": ["Train a baseline model"],
        "risk_warnings": [],
        "evidence_usage": ["Use judged course evidence"],
    }


def test_registry_is_independent_and_covers_all_resource_types():
    assert set(RESOURCE_VALIDATORS) == {
        "mindmap",
        "quiz",
        "review_doc",
        "code_practice",
        "video_script",
        "video_animation",
        "study_plan",
    }


@pytest.mark.parametrize(
    ("resource_type", "artifact", "state_updates"),
    [
        (
            "mindmap",
            {
                "title": "ML map",
                "tree": {
                    "title": "Machine learning",
                    "children": [{"title": "Models"}],
                },
            },
            {},
        ),
        ("quiz", {"title": "Quiz"}, {"exercise_items": [{"question": "Q1"}]}),
        ("review_doc", {"title": "Review", "markdown": "# Review"}, {}),
        (
            "code_practice",
            {"title": "Code", "markdown": "```python\nprint(1)\n```"},
            {},
        ),
        (
            "video_script",
            {"title": "Script", "markdown": "# Script"},
            {},
        ),
        ("study_plan", _study_plan(), {"study_plan_markdown": "# Plan"}),
    ],
)
def test_inline_resources_require_real_business_content(
    artifact_roots: dict[str, Path],
    resource_type: str,
    artifact: dict,
    state_updates: dict,
):
    result = validate_renderable_resource_result(
        resource_type,
        artifact,
        [],
        state_updates,
    )

    assert result.valid is True
    assert result.terminal_status == "success"
    assert result.renderable_count >= 1


def test_empty_and_root_only_resources_are_not_success(
    artifact_roots: dict[str, Path],
):
    empty = validate_renderable_resource_result("review_doc", {}, [], {})
    root_only = validate_renderable_resource_result(
        "mindmap",
        {"tree": {"title": "Only root", "children": []}},
        [],
        {},
    )

    assert empty.terminal_status == "failed"
    assert root_only.terminal_status == "failed"


def test_verified_local_file_and_remote_only_reference_have_distinct_truth(
    artifact_roots: dict[str, Path],
):
    local = _local_artifact(
        artifact_roots["review_doc"],
        artifact_id="artifact-1",
        filename="review.md",
        url_field="markdown_url",
        filename_field="filename",
    )
    local_result = validate_renderable_resource_result(
        "review_doc", {"title": "Review", **local}, [], {}
    )
    remote_result = validate_renderable_resource_result(
        "review_doc",
        {"title": "Review", "markdown_url": "https://example.com/review.md"},
        [],
        {},
    )

    assert local_result.terminal_status == "success"
    assert local_result.verified_local_count == 1
    assert remote_result.terminal_status == "partial_success"
    assert remote_result.remote_unverified_count == 1


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.com/review.md",
        "https://example.com/review.md?token=secret",
        "https://example.com/review.md?X-Amz-Signature=secret",
        "file:///C:/Users/example/review.md",
    ],
)
def test_unsafe_remote_references_are_rejected(
    artifact_roots: dict[str, Path],
    url: str,
):
    result = validate_renderable_resource_result(
        "review_doc", {"title": "Review", "markdown_url": url}, [], {}
    )

    assert result.terminal_status == "failed"
    assert result.downloadable_count == 0


def test_local_path_escape_zero_file_and_wrong_suffix_are_rejected(
    artifact_roots: dict[str, Path],
):
    root = artifact_roots["review_doc"]
    escaped = root.parent / "outside" / "review.md"
    escaped.parent.mkdir(parents=True)
    escaped.write_text("outside", encoding="utf-8")
    zero = _local_artifact(
        root,
        artifact_id="zero",
        filename="review.md",
        url_field="markdown_url",
        filename_field="filename",
        body=b"",
    )
    wrong = _local_artifact(
        root,
        artifact_id="wrong",
        filename="review.exe",
        url_field="markdown_url",
        filename_field="filename",
    )

    escaped_result = validate_renderable_resource_result(
        "review_doc",
        {
            "title": "Review",
            "artifact_id": "../outside",
            "filename": "review.md",
            "markdown_url": "/artifacts/outside/review.md",
        },
        [],
        {},
    )
    zero_result = validate_renderable_resource_result(
        "review_doc", {"title": "Review", **zero}, [], {}
    )
    wrong_result = validate_renderable_resource_result(
        "review_doc", {"title": "Review", **wrong}, [], {}
    )

    assert escaped_result.terminal_status == "failed"
    assert zero_result.terminal_status == "failed"
    assert wrong_result.terminal_status == "failed"


def test_video_animation_has_three_terminal_levels(
    artifact_roots: dict[str, Path],
):
    root = artifact_roots["video_animation"]
    mp4 = _local_artifact(
        root,
        artifact_id="video-ok",
        filename="lesson.mp4",
        url_field="mp4_url",
        filename_field="mp4_filename",
    )
    html = _local_artifact(
        root,
        artifact_id="video-preview",
        filename="lesson.html",
        url_field="html_url",
        filename_field="html_filename",
    )

    success = validate_renderable_resource_result(
        "video_animation", {"title": "Lesson", "render_success": True, **mp4}, [], {}
    )
    partial = validate_renderable_resource_result(
        "video_animation", {"title": "Lesson", "render_success": False, **html}, [], {}
    )
    failed = validate_renderable_resource_result(
        "video_animation", {"title": "Lesson", "render_success": False}, [], {}
    )

    assert success.terminal_status == "success"
    assert partial.terminal_status == "partial_success"
    assert failed.terminal_status == "failed"
