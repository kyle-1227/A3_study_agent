"""Strict failure semantics for code-practice resource generation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from src.config import load_settings
from src.config.evidence_orchestration_config import (
    load_resource_evidence_profiles,
)
from src.graph.code_practice import (
    CodePracticeApprovalError,
    CodePracticeGenerationError,
    CodePracticeReviewError,
    CodePracticeReviewVerdict,
    _code_practice_model_name,
    _code_practice_reviewer_model_name,
    code_practice_agent,
    code_practice_output,
    code_practice_reviewer,
    should_rewrite_code_practice,
)
from src.llm.structured_output import StructuredLLMResult, StructuredOutputError


VALID_MARKDOWN = """# Python 成绩统计实操

## 一、实操目标
使用函数和类统计学生成绩，并输出清晰报告。

## 二、前置知识
- Python 列表与循环
- 函数和类的基本语法

## 三、案例场景
读取一组学生成绩，计算平均分并列出及格学生。

## 四、完整代码
```python
class GradeReport:
    def __init__(self, scores):
        self.scores = scores

    def average(self):
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def passed(self):
        return [score for score in self.scores if score >= 60]


def main():
    report = GradeReport([92, 76, 58, 81])
    print(f"平均分: {report.average():.1f}")
    print(f"及格成绩: {report.passed()}")


if __name__ == "__main__":
    main()
```

## 五、代码逐段讲解
`GradeReport` 封装成绩数据，`average` 和 `passed` 分别计算平均分与筛选及格成绩。

## 六、运行方式
保存为 `grade_report.py`，在终端执行 `python grade_report.py`。

## 七、预期输出
程序输出平均分和及格成绩列表。

## 八、常见错误与排查
如果出现语法错误，请检查缩进、括号和字符串引号是否配对。

## 九、拓展任务
增加最高分、最低分和成绩分段统计。

## 十、自测问题
为什么空成绩列表需要单独处理？
"""


def _state(**overrides: object) -> dict:
    state = {
        "messages": [HumanMessage(content="生成 Python 成绩统计代码实操")],
        "primary_subject": "python",
        "context": [{"source": "python.md", "content": "Python score notes"}],
        "code_practice_outline": "Python score report with a class and functions",
        "code_practice_markdown": VALID_MARKDOWN,
        "code_practice_review_verdict": "approve",
        "code_practice_review_reason": "结构、代码和教学说明均通过。",
        "code_practice_round": 1,
    }
    state.update(overrides)
    return state


def _failed_result() -> StructuredLLMResult:
    return StructuredLLMResult(
        success=False,
        parsed=None,
        node_name="code_practice_reviewer",
        llm_node="code_practice_reviewer",
        schema_name="CodePracticeReviewVerdict",
        provider="test",
        model="test",
        output_mode="deepseek_tool_call_strict",
        failure_phase="business_validation_error",
        error_type="BusinessValidationError",
        error_message="invalid verdict",
    )


def test_code_practice_verdict_is_strict() -> None:
    with pytest.raises(ValidationError):
        CodePracticeReviewVerdict.model_validate(
            {"verdict": "approve", "reason": "ok", "unexpected": True}
        )

    with pytest.raises(ValidationError):
        CodePracticeReviewVerdict.model_validate({"verdict": "approve", "reason": 123})


def test_code_practice_model_requires_explicit_config() -> None:
    with (
        patch("src.graph.code_practice.get_setting", return_value=None),
        pytest.raises(ValueError, match="explicitly configured"),
    ):
        _code_practice_model_name()


def test_code_practice_reviewer_model_requires_explicit_config() -> None:
    with (
        patch("src.graph.code_practice.get_setting", return_value=None),
        pytest.raises(ValueError, match="explicitly configured"),
    ):
        _code_practice_reviewer_model_name()


def test_code_practice_runtime_configuration_is_explicit() -> None:
    settings = load_settings(reload=True)
    generation_config = settings["llm"]["code_practice"]
    reviewer_config = settings["llm"]["code_practice_reviewer"]

    assert generation_config == {
        "provider": "deepseek_official",
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "beta_base_url": "https://api.deepseek.com/beta",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.2,
        "max_generation_rounds": 2,
        "max_tokens": 3072,
        "thinking": "disabled",
        "streaming": True,
    }
    assert reviewer_config == {
        "provider": "deepseek_official",
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "beta_base_url": "https://api.deepseek.com/beta",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.0,
        "max_tokens": 1024,
        "thinking": "disabled",
        "streaming": False,
    }
    assert settings["llm_outputs"]["code_practice_reviewer"]["output_mode"] == (
        "deepseek_tool_call_strict"
    )


def test_code_practice_evidence_profile_keeps_semantics_required() -> None:
    profiles = load_resource_evidence_profiles(
        Path("config/rag/resource_evidence_profiles.yaml")
    )
    profile = profiles.profile_for("code_practice")

    assert [(need.need_id, need.criticality) for need in profile.needs] == [
        ("api_semantics", "required"),
        ("executable_patterns", "supporting"),
    ]
    assert "unsupported operations must be omitted" in (
        profile.needs[0].acceptance_criteria
    )


def test_code_practice_legacy_fallback_symbols_are_removed() -> None:
    graph_source = Path("src/graph/code_practice.py").read_text(encoding="utf-8")
    tool_source = Path("src/tools/document_tool.py").read_text(encoding="utf-8")

    for forbidden in (
        "_fallback_code_practice_markdown",
        "llm_fallback_used",
        "quality_warning",
        'get_setting("code_practice.temperature"',
        'get_setting("code_practice.model"',
    ):
        assert forbidden not in graph_source
    assert "create_code_practice_artifact" not in tool_source
    assert 'print("请在 Markdown 文档中查看代码实操内容")' not in tool_source


@pytest.mark.anyio
async def test_code_practice_agent_blocks_insufficient_evidence_without_provider() -> (
    None
):
    provider = AsyncMock()
    with (
        patch("src.graph.code_practice.invoke_plain_llm_fail_fast", provider),
        pytest.raises(CodePracticeGenerationError, match="evidence is insufficient"),
    ):
        await code_practice_agent(
            _state(degraded_generation=True, evidence_judge_state="insufficient")
        )

    provider.assert_not_awaited()


@pytest.mark.anyio
async def test_code_practice_agent_propagates_provider_failure() -> None:
    provider_error = ConnectionError("code practice provider failed")
    with (
        patch(
            "src.graph.code_practice.invoke_plain_llm_fail_fast",
            side_effect=provider_error,
        ),
        pytest.raises(ConnectionError) as exc_info,
    ):
        await code_practice_agent(_state())

    assert exc_info.value is provider_error


@pytest.mark.anyio
async def test_code_practice_reviewer_rejects_failed_structured_result() -> None:
    failed_result = _failed_result()
    with (
        patch(
            "src.graph.code_practice.invoke_structured_llm",
            return_value=failed_result,
        ),
        pytest.raises(StructuredOutputError) as exc_info,
    ):
        await code_practice_reviewer(_state())

    assert exc_info.value.result is failed_result


@pytest.mark.anyio
async def test_code_practice_reviewer_propagates_provider_failure() -> None:
    provider_error = ConnectionError("reviewer provider failed")
    with (
        patch(
            "src.graph.code_practice.invoke_structured_llm",
            side_effect=provider_error,
        ),
        pytest.raises(ConnectionError) as exc_info,
    ):
        await code_practice_reviewer(_state())

    assert exc_info.value is provider_error


@pytest.mark.anyio
async def test_code_practice_reviewer_rejects_unexpected_parsed_type() -> None:
    with (
        patch(
            "src.graph.code_practice.invoke_structured_llm",
            return_value=SimpleNamespace(success=True, parsed=object()),
        ),
        pytest.raises(CodePracticeReviewError, match="parsed result"),
    ):
        await code_practice_reviewer(_state())


@pytest.mark.anyio
async def test_code_practice_reviewer_preserves_real_reject() -> None:
    verdict = CodePracticeReviewVerdict(
        verdict="reject", reason="The exercise is fundamentally unusable."
    )
    with patch(
        "src.graph.code_practice.invoke_structured_llm",
        return_value=SimpleNamespace(success=True, parsed=verdict),
    ) as reviewer:
        result = await code_practice_reviewer(_state())

    assert result["code_practice_review_verdict"] == "reject"
    assert result["code_practice_revision_notes"] == verdict.reason
    assert reviewer.await_args.kwargs["llm_node"] == "code_practice_reviewer"


@pytest.mark.anyio
@pytest.mark.parametrize("verdict", ["", "revise", "reject", "unexpected"])
async def test_code_practice_output_requires_explicit_approve(verdict: str) -> None:
    writer = Mock()
    with (
        patch("src.graph.code_practice.create_document_artifact", writer),
        pytest.raises(CodePracticeApprovalError, match="approve verdict"),
    ):
        await code_practice_output(_state(code_practice_review_verdict=verdict))

    writer.assert_not_called()


@pytest.mark.anyio
async def test_code_practice_output_revalidates_markdown_before_io() -> None:
    writer = Mock()
    with (
        patch("src.graph.code_practice.create_document_artifact", writer),
        pytest.raises(CodePracticeApprovalError, match="local quality check"),
    ):
        await code_practice_output(
            _state(code_practice_markdown="# Python\n\n```python\nprint('x')\n```")
        )

    writer.assert_not_called()


@pytest.mark.anyio
async def test_code_practice_output_propagates_document_artifact_failure() -> None:
    artifact_error = OSError("artifact storage unavailable")
    with (
        patch(
            "src.graph.code_practice.create_document_artifact",
            side_effect=artifact_error,
        ),
        pytest.raises(OSError) as exc_info,
    ):
        await code_practice_output(_state())

    assert exc_info.value is artifact_error


@pytest.mark.anyio
async def test_code_practice_output_writes_real_provider_code(tmp_path: Path) -> None:
    document_artifact = {
        "artifact_id": "code-practice-1",
        "filename": "grade-report.md",
        "markdown_url": "/grade-report.md",
        "docx_url": "/grade-report.docx",
    }
    with (
        patch(
            "src.graph.code_practice.create_document_artifact",
            return_value=document_artifact,
        ),
        patch(
            "src.graph.code_practice.get_code_practice_artifact_dir",
            return_value=tmp_path,
        ),
    ):
        result = await code_practice_output(_state())

    python_path = tmp_path / "code-practice-1" / "grade-report.py"
    assert python_path.is_file()
    assert "class GradeReport" in python_path.read_text(encoding="utf-8")
    assert result["code_practice_artifact"]["python_url"].endswith("/grade-report.py")
    assert "quality_warning" not in result["code_practice_artifact"]
    assert isinstance(result["messages"][0], AIMessage)


def test_code_practice_router_blocks_unknown_or_exhausted_verdict() -> None:
    with pytest.raises(CodePracticeApprovalError, match="requires approve"):
        should_rewrite_code_practice(_state(code_practice_review_verdict=""))

    with pytest.raises(CodePracticeApprovalError, match="maximum rewrite rounds"):
        should_rewrite_code_practice(
            _state(code_practice_review_verdict="reject", code_practice_round=2)
        )


def test_code_practice_router_allows_only_approved_output() -> None:
    assert should_rewrite_code_practice(_state()) == "output"
    assert (
        should_rewrite_code_practice(
            _state(code_practice_review_verdict="revise", code_practice_round=1)
        )
        == "rewrite"
    )
