"""Strict failure semantics for teaching-video script generation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from src.config import load_settings
from src.graph.video_script import (
    VideoScriptApprovalError,
    VideoScriptGenerationError,
    VideoScriptReviewVerdict,
    _video_script_model_name,
    should_rewrite_video_script,
    video_script_agent,
    video_script_output,
    video_script_reviewer,
)
from src.llm.structured_output import StructuredLLMResult, StructuredOutputError


VALID_MARKDOWN = """# Python 循环教学视频

## 一、视频基本信息
- 主题：Python 循环
- 目标：理解 for 与 while

## 二、知识点拆解
- for 循环遍历序列
- while 循环依据条件执行

## 三、视频分镜脚本
| 镜头 | 时间 | 画面内容 | 旁白 | 字幕 | 动画说明 |
|---|---|---|---|---|---|
| 1 | 00:00-00:20 | Python 编辑器 | 本节学习循环结构。 | Python 循环 | 代码逐行高亮 |

## 四、完整旁白文案
Python 循环用于重复执行任务。for 循环适合遍历已知序列，while 循环适合在条件成立期间继续执行，并且必须注意退出条件。

## 五、字幕 SRT
1
00:00:00,000 --> 00:00:05,000
本节学习 Python 循环结构。

## 六、动画设计说明
编辑器中的循环语句逐行高亮，变量值在右侧面板同步变化，退出条件使用醒目标记展示。

## 七、板书内容
- for：遍历序列
- while：条件循环

## 八、互动提问
什么时候应该选择 while 循环？

## 九、结尾总结
根据任务是否存在明确序列和退出条件选择循环结构。

## 十、拓展练习
编写循环统计一组成绩中的及格人数。
"""


def _state(**overrides: object) -> dict:
    state = {
        "messages": [HumanMessage(content="制作一个 Python 循环教学视频")],
        "primary_subject": "python",
        "context": [{"source": "python.md", "content": "Python loop notes"}],
        "video_script_outline": "Python for and while loops",
        "video_script_markdown": VALID_MARKDOWN,
        "video_script_review_verdict": "approve",
        "video_script_review_reason": "结构和教学内容均通过。",
        "video_script_round": 1,
    }
    state.update(overrides)
    return state


def _failed_result() -> StructuredLLMResult:
    return StructuredLLMResult(
        success=False,
        parsed=None,
        node_name="video_script_reviewer",
        llm_node="video_script",
        schema_name="VideoScriptReviewVerdict",
        provider="test",
        model="test",
        output_mode="deepseek_tool_call_strict",
        failure_phase="business_validation_error",
        error_type="BusinessValidationError",
        error_message="invalid verdict",
    )


def test_video_script_verdict_forbids_extra_fields_and_coercion() -> None:
    with pytest.raises(ValidationError):
        VideoScriptReviewVerdict.model_validate(
            {"verdict": "approve", "reason": "ok", "unexpected": True}
        )

    with pytest.raises(ValidationError):
        VideoScriptReviewVerdict.model_validate({"verdict": "approve", "reason": 123})


def test_video_script_model_requires_explicit_config() -> None:
    with (
        patch("src.graph.video_script.get_setting", return_value=None),
        pytest.raises(ValueError, match="explicitly configured"),
    ):
        _video_script_model_name()


def test_video_script_runtime_configuration_is_explicit() -> None:
    settings = load_settings(reload=True)

    assert settings["llm"]["video_script"] == {
        "provider": "deepseek_official",
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "beta_base_url": "https://api.deepseek.com/beta",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.2,
        "max_tokens": 4096,
        "thinking": "disabled",
        "streaming": False,
    }
    assert settings["llm_outputs"]["video_script_reviewer"]["output_mode"] == (
        "deepseek_tool_call_strict"
    )


def test_video_script_legacy_fallback_symbols_are_removed() -> None:
    source = Path("src/graph/video_script.py").read_text(encoding="utf-8")

    for forbidden in (
        "_fallback_video_script_markdown",
        "_fallback_video_script_outline",
        "VIDEO_SCRIPT_DEFAULT_MODEL",
        "llm_fallback_used",
        "quality_warning",
    ):
        assert forbidden not in source


@pytest.mark.anyio
async def test_video_script_agent_blocks_insufficient_evidence_without_provider() -> (
    None
):
    provider = AsyncMock()
    with (
        patch("src.graph.video_script.invoke_plain_llm_fail_fast", provider),
        pytest.raises(VideoScriptGenerationError, match="evidence is insufficient"),
    ):
        await video_script_agent(
            _state(degraded_generation=True, evidence_judge_state="insufficient")
        )

    provider.assert_not_awaited()


@pytest.mark.anyio
async def test_video_script_agent_propagates_provider_failure() -> None:
    provider_error = ConnectionError("video script provider failed")
    with (
        patch(
            "src.graph.video_script.invoke_plain_llm_fail_fast",
            side_effect=provider_error,
        ),
        pytest.raises(ConnectionError) as exc_info,
    ):
        await video_script_agent(_state())

    assert exc_info.value is provider_error


@pytest.mark.anyio
async def test_video_script_reviewer_rejects_failed_structured_result() -> None:
    failed_result = _failed_result()
    with (
        patch(
            "src.graph.video_script.invoke_structured_llm",
            return_value=failed_result,
        ),
        pytest.raises(StructuredOutputError) as exc_info,
    ):
        await video_script_reviewer(_state())

    assert exc_info.value.result is failed_result


@pytest.mark.anyio
async def test_video_script_reviewer_propagates_provider_failure() -> None:
    provider_error = ConnectionError("reviewer provider failed")
    with (
        patch(
            "src.graph.video_script.invoke_structured_llm",
            side_effect=provider_error,
        ),
        pytest.raises(ConnectionError) as exc_info,
    ):
        await video_script_reviewer(_state())

    assert exc_info.value is provider_error


@pytest.mark.anyio
async def test_video_script_reviewer_preserves_real_reject() -> None:
    verdict = VideoScriptReviewVerdict(
        verdict="reject", reason="旁白需要更清晰地解释退出条件。"
    )
    with patch(
        "src.graph.video_script.invoke_structured_llm",
        return_value=SimpleNamespace(success=True, parsed=verdict),
    ):
        result = await video_script_reviewer(_state())

    assert result["video_script_review_verdict"] == "reject"
    assert result["video_script_revision_notes"] == verdict.reason


@pytest.mark.anyio
@pytest.mark.parametrize("verdict", ["", "reject", "unexpected"])
async def test_video_script_output_requires_explicit_approve(verdict: str) -> None:
    artifact_writer = Mock()
    with (
        patch("src.graph.video_script._create_video_script_artifact", artifact_writer),
        pytest.raises(VideoScriptApprovalError, match="approve verdict"),
    ):
        await video_script_output(_state(video_script_review_verdict=verdict))

    artifact_writer.assert_not_called()


@pytest.mark.anyio
async def test_video_script_output_does_not_replace_invalid_markdown() -> None:
    artifact_writer = Mock()
    with (
        patch("src.graph.video_script._create_video_script_artifact", artifact_writer),
        pytest.raises(VideoScriptApprovalError, match="local quality check"),
    ):
        await video_script_output(_state(video_script_markdown="# Python"))

    artifact_writer.assert_not_called()


@pytest.mark.anyio
async def test_video_script_output_propagates_artifact_failure() -> None:
    artifact_error = OSError("artifact storage unavailable")
    with (
        patch(
            "src.graph.video_script._create_video_script_artifact",
            side_effect=artifact_error,
        ),
        pytest.raises(OSError) as exc_info,
    ):
        await video_script_output(_state())

    assert exc_info.value is artifact_error


@pytest.mark.anyio
async def test_video_script_output_emits_only_real_approved_artifact() -> None:
    artifact = {
        "artifact_id": "video-script-1",
        "markdown_url": "/artifacts/video-script-1.md",
        "docx_url": "/artifacts/video-script-1.docx",
        "srt_url": "/artifacts/video-script-1.srt",
    }
    with patch(
        "src.graph.video_script._create_video_script_artifact",
        return_value=artifact,
    ):
        result = await video_script_output(_state())

    assert result["video_script_artifact"]["artifact_id"] == "video-script-1"
    assert "quality_warning" not in result["video_script_artifact"]
    assert isinstance(result["messages"][0], AIMessage)


def test_video_script_router_blocks_unknown_or_exhausted_verdict() -> None:
    with pytest.raises(VideoScriptApprovalError, match="explicit approve or reject"):
        should_rewrite_video_script(_state(video_script_review_verdict=""))

    with pytest.raises(VideoScriptApprovalError, match="maximum rewrite rounds"):
        should_rewrite_video_script(
            _state(video_script_review_verdict="reject", video_script_round=2)
        )


def test_video_script_router_allows_only_approved_output() -> None:
    assert should_rewrite_video_script(_state()) == "output"
    assert (
        should_rewrite_video_script(
            _state(video_script_review_verdict="reject", video_script_round=1)
        )
        == "rewrite"
    )
