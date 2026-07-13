"""Strict contracts and terminal truth for rendered teaching animations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from src.config import load_settings
from src.graph.video_animation import (
    VideoAnimationApprovalError,
    VideoAnimationGenerationError,
    VideoAnimationRenderError,
    _video_animation_model_name,
    should_rewrite_video_animation,
    video_animation_agent,
    video_animation_output,
    video_animation_planner,
    video_animation_reviewer,
)
from src.llm.structured_output import StructuredLLMResult, StructuredOutputError
from src.tools.video_animation_contracts import (
    AnimationReviewVerdictV1,
    VideoAnimationSpecV1,
    validate_video_animation_spec,
)
from src.tools.video_animation_tool import create_video_animation_artifact_async


REQUIRED_STEPS = [
    "fade_in",
    "move",
    "highlight",
    "arrow_draw",
    "code_highlight",
    "fade_out",
]


def _spec_payload() -> dict:
    scenes = []
    for index in range(5):
        start = float(index * 5)
        scenes.append(
            {
                "scene_id": f"scene_{index + 1}",
                "start": start,
                "end": start + 5.0,
                "title": f"Python 循环场景 {index + 1}",
                "subtitle": "展示循环执行过程",
                "narration": "通过变量变化说明 Python 循环如何逐步执行。",
                "visual_type": "concept_diagram",
                "elements": [
                    {
                        "type": "box",
                        "text": f"循环步骤 {index + 1}",
                        "x": 120.0,
                        "y": 180.0,
                        "width": 260.0,
                        "height": 80.0,
                    }
                ],
                "animation_steps": list(REQUIRED_STEPS),
            }
        )
    return {
        "schema_version": "video_animation_spec_v1",
        "title": "Python 循环教学动画",
        "duration_seconds": 25.0,
        "resolution": {"width": 1280, "height": 720},
        "style": {
            "theme": "clean academic",
            "background": "#f8fafc",
            "font": "Microsoft YaHei, Arial, sans-serif",
        },
        "scenes": scenes,
    }


def _spec() -> VideoAnimationSpecV1:
    return VideoAnimationSpecV1.model_validate(_spec_payload())


def _state(**overrides: object) -> dict:
    state = {
        "messages": [HumanMessage(content="制作一个 Python 循环教学动画")],
        "primary_subject": "python",
        "context": [{"source": "python.md", "content": "Python loop notes"}],
        "video_animation_spec": _spec_payload(),
        "video_animation_review_verdict": "approve",
        "video_animation_review_reason": "教学结构通过。",
        "video_animation_round": 1,
    }
    state.update(overrides)
    return state


def _failed_result(node_name: str, schema_name: str) -> StructuredLLMResult:
    return StructuredLLMResult(
        success=False,
        parsed=None,
        node_name=node_name,
        llm_node="video_animation",
        schema_name=schema_name,
        provider="test",
        model="test",
        output_mode="deepseek_tool_call_strict",
        failure_phase="business_validation_error",
        error_type="BusinessValidationError",
        error_message="invalid animation contract",
    )


def _artifact(*, render_success: bool, render_mode: str = "production") -> dict:
    formal = render_success and render_mode == "production"
    return {
        "artifact_id": "animation-1",
        "title": "Python 循环教学动画",
        "html_filename": "animation.html",
        "json_filename": "animation.json",
        "srt_filename": "animation.srt",
        "mp4_filename": "animation.mp4" if render_success else "",
        "html_url": "/artifacts/video-animations/animation-1/animation.html",
        "json_url": "/artifacts/video-animations/animation-1/animation.json",
        "srt_url": "/artifacts/video-animations/animation-1/animation.srt",
        "mp4_url": (
            "/artifacts/video-animations/animation-1/animation.mp4"
            if render_success
            else ""
        ),
        "render_mode": render_mode,
        "is_preview_video": render_mode == "test",
        "video_valid_for_teaching": formal,
        "html_available": True,
        "json_available": True,
        "srt_available": True,
        "mp4_available": render_success,
        "mp4_exists": render_success,
        "mp4_file_size": 4096 if render_success else 0,
        "render_success": render_success,
        "render_log": "render complete" if render_success else "ffmpeg unavailable",
        "full_duration_seconds": 30,
        "render_duration_seconds": 30 if formal else 0,
        "fps": 24 if render_mode == "production" else 12,
        "frame_count": 600 if render_success else 0,
        "ffmpeg_path": "ffmpeg" if render_success else "",
        "playwright_available": True,
    }


def test_video_animation_spec_forbids_extra_and_alias_fields() -> None:
    payload = _spec_payload()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        VideoAnimationSpecV1.model_validate(payload)

    payload = _spec_payload()
    scene = payload["scenes"][0]
    scene["voiceover"] = scene.pop("narration")
    with pytest.raises(ValidationError):
        VideoAnimationSpecV1.model_validate(payload)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda payload: payload.update(scenes=payload["scenes"][:4]), "5"),
        (
            lambda payload: payload["scenes"][1].update(start=4.0),
            "overlaps",
        ),
        (
            lambda payload: payload["scenes"][1].update(
                scene_id=payload["scenes"][0]["scene_id"]
            ),
            "unique",
        ),
        (
            lambda payload: payload["scenes"][0].update(
                animation_steps=REQUIRED_STEPS[:-1]
            ),
            "6",
        ),
    ],
)
def test_video_animation_spec_rejects_invalid_structure(mutate, expected: str) -> None:
    payload = _spec_payload()
    mutate(payload)
    try:
        parsed = VideoAnimationSpecV1.model_validate(payload)
    except ValidationError as exc:
        assert expected in str(exc)
    else:
        assert expected in validate_video_animation_spec(parsed)


def test_video_animation_model_requires_explicit_config() -> None:
    with (
        patch("src.graph.video_animation.get_setting", return_value=None),
        pytest.raises(ValueError, match="explicitly configured"),
    ):
        _video_animation_model_name()


def test_video_animation_runtime_configuration_is_explicit() -> None:
    settings = load_settings(reload=True)
    llm_config = settings["llm"]["video_animation"]

    assert llm_config["provider"] == "deepseek_official"
    assert llm_config["model"] == "deepseek-v4-pro"
    assert llm_config["temperature"] == 0.2
    assert settings["video_animation"] == {
        "render_mode": "production",
        "max_duration_seconds": 90,
        "max_generation_rounds": 2,
    }
    for node_name in (
        "video_animation_planner",
        "video_animation_agent",
        "video_animation_reviewer",
    ):
        assert settings["llm_outputs"][node_name]["output_mode"] == (
            "deepseek_tool_call_strict"
        )


def test_video_animation_legacy_fallback_symbols_are_removed() -> None:
    graph_source = Path("src/graph/video_animation.py").read_text(encoding="utf-8")
    tool_source = Path("src/tools/video_animation_tool.py").read_text(encoding="utf-8")

    for forbidden in (
        "_fallback_animation_spec",
        "_parse_json_object",
        "_ensure_animation_spec",
        "fallback_used",
        "planner fallback seed",
    ):
        assert forbidden not in graph_source
    for forbidden in (
        "_fallback_scenes",
        "create_video_animation_artifact(",
        "Please view the animation HTML",
    ):
        assert forbidden not in tool_source


@pytest.mark.anyio
async def test_video_animation_planner_rejects_failed_structured_result() -> None:
    failed_result = _failed_result("video_animation_planner", "VideoAnimationSpecV1")
    with (
        patch(
            "src.graph.video_animation.invoke_structured_llm",
            return_value=failed_result,
        ),
        pytest.raises(StructuredOutputError) as exc_info,
    ):
        await video_animation_planner(_state())

    assert exc_info.value.result is failed_result


@pytest.mark.anyio
async def test_video_animation_agent_blocks_insufficient_evidence_without_provider() -> (
    None
):
    provider = AsyncMock()
    with (
        patch("src.graph.video_animation.invoke_structured_llm", provider),
        pytest.raises(VideoAnimationGenerationError, match="evidence is insufficient"),
    ):
        await video_animation_agent(
            _state(degraded_generation=True, evidence_judge_state="insufficient")
        )

    provider.assert_not_awaited()


@pytest.mark.anyio
async def test_video_animation_agent_propagates_provider_failure() -> None:
    provider_error = ConnectionError("animation provider failed")
    with (
        patch(
            "src.graph.video_animation.invoke_structured_llm",
            side_effect=provider_error,
        ),
        pytest.raises(ConnectionError) as exc_info,
    ):
        await video_animation_agent(_state())

    assert exc_info.value is provider_error


@pytest.mark.anyio
async def test_video_animation_reviewer_local_failure_skips_llm() -> None:
    reviewer = AsyncMock()
    invalid = _spec_payload()
    invalid["scenes"] = invalid["scenes"][:2]
    with patch("src.graph.video_animation.invoke_structured_llm", reviewer):
        result = await video_animation_reviewer(_state(video_animation_spec=invalid))

    reviewer.assert_not_awaited()
    assert result["video_animation_review_verdict"] == "reject"


@pytest.mark.anyio
async def test_video_animation_reviewer_requires_real_structured_approval() -> None:
    verdict = AnimationReviewVerdictV1(
        verdict="approve", reason="Scene progression is clear and teachable."
    )
    with patch(
        "src.graph.video_animation.invoke_structured_llm",
        return_value=SimpleNamespace(success=True, parsed=verdict),
    ) as reviewer:
        result = await video_animation_reviewer(_state())

    reviewer.assert_awaited_once()
    assert result["video_animation_review_verdict"] == "approve"


@pytest.mark.anyio
async def test_video_animation_reviewer_rejects_failed_structured_result() -> None:
    failed_result = _failed_result(
        "video_animation_reviewer", "AnimationReviewVerdictV1"
    )
    with (
        patch(
            "src.graph.video_animation.invoke_structured_llm",
            return_value=failed_result,
        ),
        pytest.raises(StructuredOutputError) as exc_info,
    ):
        await video_animation_reviewer(_state())

    assert exc_info.value.result is failed_result


@pytest.mark.anyio
async def test_animation_tool_rejects_invalid_spec_before_artifact_io() -> None:
    root_resolver = Mock()
    with (
        patch(
            "src.tools.video_animation_tool.get_video_animation_artifact_dir",
            root_resolver,
        ),
        pytest.raises(ValidationError),
    ):
        await create_video_animation_artifact_async(
            animation_spec={},
            title="Invalid",
            srt_text=None,
            fps=24,
            width=1280,
            height=720,
            max_duration_seconds=90,
            render_mode="production",
        )

    root_resolver.assert_not_called()


@pytest.mark.anyio
async def test_animation_tool_writes_validated_real_artifacts(tmp_path: Path) -> None:
    async def fake_renderer(**kwargs):
        kwargs["mp4_path"].write_bytes(b"real-mp4")
        return {
            "render_success": True,
            "render_log": "render complete",
            "frame_count": 720,
            "ffmpeg_path": "ffmpeg",
            "playwright_available": True,
        }

    with (
        patch(
            "src.tools.video_animation_tool.get_video_animation_artifact_dir",
            return_value=tmp_path,
        ),
        patch(
            "src.tools.video_animation_tool.render_html_animation_to_mp4_async",
            side_effect=fake_renderer,
        ),
    ):
        artifact = await create_video_animation_artifact_async(
            animation_spec=_spec_payload(),
            title="Python 循环教学动画",
            srt_text=None,
            fps=24,
            width=1280,
            height=720,
            max_duration_seconds=90,
            render_mode="production",
        )

    artifact_dir = tmp_path / artifact["artifact_id"]
    assert (artifact_dir / artifact["html_filename"]).is_file()
    assert (artifact_dir / artifact["json_filename"]).is_file()
    assert (artifact_dir / artifact["srt_filename"]).is_file()
    assert (artifact_dir / artifact["mp4_filename"]).read_bytes() == b"real-mp4"
    assert artifact["video_valid_for_teaching"] is True


@pytest.mark.anyio
async def test_animation_tool_marks_truncated_production_mp4_not_teaching_valid(
    tmp_path: Path,
) -> None:
    payload = _spec_payload()
    payload["duration_seconds"] = 60.0

    async def fake_renderer(**kwargs):
        kwargs["mp4_path"].write_bytes(b"truncated-mp4")
        return {
            "render_success": True,
            "render_log": "render complete",
            "frame_count": 720,
            "ffmpeg_path": "ffmpeg",
            "playwright_available": True,
        }

    with (
        patch(
            "src.tools.video_animation_tool.get_video_animation_artifact_dir",
            return_value=tmp_path,
        ),
        patch(
            "src.tools.video_animation_tool.render_html_animation_to_mp4_async",
            side_effect=fake_renderer,
        ),
    ):
        artifact = await create_video_animation_artifact_async(
            animation_spec=payload,
            title="Python 循环教学动画",
            srt_text=None,
            fps=24,
            width=1280,
            height=720,
            max_duration_seconds=90,
            render_mode="production",
        )

    assert artifact["render_success"] is True
    assert artifact["full_duration_seconds"] == 60
    assert artifact["render_duration_seconds"] == 30
    assert artifact["video_valid_for_teaching"] is False


@pytest.mark.anyio
@pytest.mark.parametrize("verdict", ["", "reject", "unexpected"])
async def test_video_animation_output_requires_explicit_approve(verdict: str) -> None:
    renderer = AsyncMock()
    with (
        patch(
            "src.graph.video_animation.create_video_animation_artifact_async",
            renderer,
        ),
        pytest.raises(VideoAnimationApprovalError, match="approve verdict"),
    ):
        await video_animation_output(_state(video_animation_review_verdict=verdict))

    renderer.assert_not_awaited()


@pytest.mark.anyio
async def test_video_animation_output_propagates_renderer_exception() -> None:
    render_error = OSError("renderer unavailable")
    with (
        patch(
            "src.graph.video_animation.create_video_animation_artifact_async",
            side_effect=render_error,
        ),
        pytest.raises(OSError) as exc_info,
    ):
        await video_animation_output(_state())

    assert exc_info.value is render_error


@pytest.mark.anyio
async def test_video_animation_output_blocks_zero_real_artifacts() -> None:
    artifact = _artifact(render_success=False)
    artifact["html_available"] = False
    with (
        patch(
            "src.graph.video_animation.create_video_animation_artifact_async",
            return_value=artifact,
        ),
        pytest.raises(VideoAnimationRenderError, match="html_available"),
    ):
        await video_animation_output(_state())


@pytest.mark.anyio
async def test_video_animation_output_preserves_real_html_partial() -> None:
    artifact = _artifact(render_success=False)
    with (
        patch(
            "src.graph.video_animation.create_video_animation_artifact_async",
            return_value=artifact,
        ),
        patch(
            "src.graph.video_animation._read_artifact_html",
            return_value="<html>real preview</html>",
        ),
    ):
        result = await video_animation_output(_state())

    assert result["video_animation_artifact"]["render_success"] is False
    assert "未达到完整正式视频标准" in result["messages"][0].content


@pytest.mark.anyio
async def test_video_animation_output_emits_verified_full_production_video() -> None:
    artifact = _artifact(render_success=True)
    with (
        patch(
            "src.graph.video_animation.create_video_animation_artifact_async",
            return_value=artifact,
        ),
        patch(
            "src.graph.video_animation._read_artifact_html",
            return_value="<html>real animation</html>",
        ),
    ):
        result = await video_animation_output(_state())

    assert result["video_animation_artifact"]["video_valid_for_teaching"] is True
    assert "完整正式教学动画" in result["messages"][0].content
    assert isinstance(result["messages"][0], AIMessage)


def test_video_animation_router_blocks_unknown_or_exhausted_verdict() -> None:
    with pytest.raises(VideoAnimationApprovalError, match="explicit approve or reject"):
        should_rewrite_video_animation(_state(video_animation_review_verdict=""))

    with pytest.raises(VideoAnimationApprovalError, match="maximum rewrite rounds"):
        should_rewrite_video_animation(
            _state(video_animation_review_verdict="reject", video_animation_round=2)
        )


def test_video_animation_router_allows_only_approved_output() -> None:
    assert should_rewrite_video_animation(_state()) == "output"
    assert (
        should_rewrite_video_animation(
            _state(video_animation_review_verdict="reject", video_animation_round=1)
        )
        == "rewrite"
    )
