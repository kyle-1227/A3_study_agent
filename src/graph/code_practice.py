"""Code-practice resource-generation nodes."""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_setting
from src.graph.llm import invoke_plain_llm_fail_fast
from src.graph.state import LearningState
from src.llm.structured_output import (
    StructuredOutputError,
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.observability.a3_trace import emit_a3_trace
from src.tools.document_tool import (
    create_document_artifact,
    get_code_practice_artifact_dir,
)
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


REQUIRED_CODE_PRACTICE_SECTIONS = {
    "title": r"(?m)^#\s+\S+",
    "goal": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?实操目标",
    "prerequisites": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?前置知识",
    "scenario": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?案例场景",
    "code": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?完整代码",
    "explanation": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?代码逐段讲解",
    "run": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?运行方式",
    "output": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?预期输出",
    "troubleshooting": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?常见错误(?:与排查)?",
    "extension": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?拓展任务",
    "self_check": r"(?m)^##\s*(?:[一二三四五六七八九十0-9]+[、.．]\s*)?自测问题",
}

PYTHON_FENCED_CODE_RE = re.compile(r"(?s)```(?:python|py)\s*\n(?P<code>.+?)```")

REQUIRED_CODE_PRACTICE_SECTION_NAMES = {
    "goal": "实操目标",
    "prerequisites": "前置知识",
    "scenario": "案例场景",
    "code": "完整代码",
    "explanation": "代码逐段讲解",
    "run": "运行方式",
    "output": "预期输出",
    "troubleshooting": "常见错误",
    "extension": "拓展任务",
    "self_check": "自测问题",
}

RUN_INSTRUCTION_MARKERS = (
    "运行",
    "执行",
    "命令",
    "终端",
    "命令行",
    "保存为",
    "python ",
    "python3 ",
    ".py",
)
EXPECTED_OUTPUT_MARKERS = (
    "预期输出",
    "输出结果",
    "示例输出",
    "应输出",
    "打印",
    "print",
)
ERROR_DEBUGGING_MARKERS = (
    "错误",
    "报错",
    "排查",
    "调试",
    "检查",
    "syntaxerror",
    "traceback",
    "exception",
)
EXTENSION_TASK_MARKERS = (
    "拓展",
    "扩展",
    "进阶",
    "挑战",
    "改造",
    "优化",
    "增加",
    "尝试",
    "add",
    "extend",
    "extension",
    "challenge",
    "improve",
)


class CodePracticeGenerationError(RuntimeError):
    """Raised when code-practice generation cannot produce a real provider result."""


class CodePracticeReviewError(RuntimeError):
    """Raised when code-practice review cannot establish an approval decision."""


class CodePracticeApprovalError(RuntimeError):
    """Raised when an unapproved code-practice result reaches artifact output."""


class CodePracticeReviewVerdict(BaseModel):
    """Structured teaching-quality gate output for code_practice_reviewer."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    verdict: Literal["approve", "revise", "reject"]
    reason: str = Field(min_length=1)


def validate_code_practice_verdict(parsed: BaseModel) -> str:
    if not isinstance(parsed, CodePracticeReviewVerdict):
        return "root expected CodePracticeReviewVerdict"
    if parsed.verdict not in {"approve", "revise", "reject"}:
        return "verdict must be approve, revise, or reject"
    if not str(parsed.reason or "").strip():
        return "reason must be non-empty"
    return ""


def _last_human_query(state: LearningState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _format_keypoints(state: LearningState) -> str:
    values = [
        str(item).strip() for item in state.get("keypoints", []) if str(item).strip()
    ]
    expanded = [
        str(item).strip()
        for item in state.get("expanded_keypoints", [])
        if str(item).strip()
    ]
    merged = values + [item for item in expanded if item not in values]
    return ", ".join(merged) or "No explicit keypoints."


def _format_context(context: list[dict]) -> str:
    if not context:
        return "No judged evidence is available. Do not invent citations."
    parts: list[str] = []
    for idx, item in enumerate(context[:8], 1):
        source = (
            item.get("source")
            or item.get("title")
            or item.get("url")
            or "learning material"
        )
        content = str(
            item.get("content") or item.get("snippet") or item.get("text") or ""
        )[:900]
        if content:
            parts.append(f"[{idx}] Source: {source}\n{content}")
    return "\n\n".join(parts) or "Judged evidence has no readable body."


def _extract_markdown_title(markdown: str) -> str:
    for line in markdown.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            title = match.group(1).strip()
            if title:
                return title
            break
    raise CodePracticeGenerationError("code practice Markdown title is missing")


def _extract_first_python_code_block(markdown: str) -> str:
    """Extract the first fenced Python code block from Markdown."""
    match = PYTHON_FENCED_CODE_RE.search(markdown or "")
    return match.group("code").strip() if match else ""


def _check_code_syntax(code: str) -> tuple[bool, str]:
    """Return whether Python code parses successfully and any syntax error."""
    if not str(code or "").strip():
        return False, "Python code block is empty"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno else "unknown line"
        return False, f"{exc.msg} ({location})"
    return True, ""


def _section_body(markdown: str, section_key: str) -> str:
    pattern = REQUIRED_CODE_PRACTICE_SECTIONS.get(section_key)
    if not pattern:
        return ""
    heading_match = re.search(pattern, markdown)
    if not heading_match:
        return ""
    start = heading_match.end()
    next_heading = re.search(r"(?m)^##\s+", markdown[start:])
    end = start + next_heading.start() if next_heading else len(markdown)
    return markdown[start:end].strip()


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(marker.lower() in lowered for marker in markers)


def _has_function_or_class(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )


def _topic_terms(state: dict) -> list[str]:
    values: list[str] = []
    for key in ("primary_subject", "learning_goal"):
        value = str(state.get(key) or "").strip()
        if value:
            values.append(value)
    values.append(_last_human_query(state))
    values.extend(str(item) for item in state.get("keypoints", []) if str(item).strip())
    values.extend(
        str(item) for item in state.get("expanded_keypoints", []) if str(item).strip()
    )

    stopwords = {
        "帮我",
        "生成",
        "一份",
        "代码",
        "实操",
        "案例",
        "完整",
        "可运行",
        "练习",
        "项目",
        "python",
    }
    terms: list[str] = []
    joined = " ".join(values).lower()
    for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_+#.-]{1,}", joined):
        if term not in terms:
            terms.append(term)
    for phrase in (
        "面向对象",
        "函数",
        "类",
        "爬虫",
        "数据分析",
        "机器学习",
        "文件操作",
        "异常处理",
    ):
        if phrase in joined and phrase not in terms:
            terms.append(phrase)
    return [term for term in terms if term not in stopwords]


def _is_topic_relevant(markdown: str, code: str, state: dict) -> bool:
    subject = str(state.get("primary_subject") or "").strip().lower()
    combined = f"{markdown}\n{code}".lower()
    if subject in {"", "other"}:
        terms = _topic_terms(state)
        if not terms:
            return False
        return any(term.lower() in combined for term in terms)
    if subject == "python":
        return bool(code.strip())
    subject_terms = {subject, subject.replace("_", " "), subject.replace("_", "")}
    return any(term and term in combined for term in subject_terms)


def _local_check_code_practice(markdown: str, state: dict) -> dict:
    """Run deterministic quality checks for a code-practice Markdown document."""
    text = str(markdown or "").strip()
    code = _extract_first_python_code_block(text)
    syntax_ok, syntax_error = _check_code_syntax(code)
    code_lines = code.splitlines()

    missing_sections = [
        name
        for key, name in REQUIRED_CODE_PRACTICE_SECTION_NAMES.items()
        if not re.search(REQUIRED_CODE_PRACTICE_SECTIONS[key], text)
    ]
    title_missing = not bool(re.search(REQUIRED_CODE_PRACTICE_SECTIONS["title"], text))
    run_body = _section_body(text, "run")
    output_body = _section_body(text, "output")
    troubleshooting_body = _section_body(text, "troubleshooting")
    extension_body = _section_body(text, "extension")

    has_code_block = bool(code)
    has_enough_code = len(code_lines) >= 15
    has_function_or_class = _has_function_or_class(code)
    has_run_instruction = bool(run_body) and _contains_any(
        run_body, RUN_INSTRUCTION_MARKERS
    )
    has_expected_output = bool(output_body)
    has_error_debugging = bool(troubleshooting_body) and _contains_any(
        troubleshooting_body,
        ERROR_DEBUGGING_MARKERS,
    )
    has_extension_tasks = bool(extension_body) and _contains_any(
        extension_body, EXTENSION_TASK_MARKERS
    )
    topic_relevant = _is_topic_relevant(text, code, state)

    failed_reasons: list[str] = []
    if not text:
        failed_reasons.append("代码实操 Markdown 为空")
    if title_missing:
        failed_reasons.append("缺少一级标题")
    if missing_sections:
        failed_reasons.append(f"缺少必要章节: {', '.join(missing_sections)}")
    if not has_code_block:
        failed_reasons.append("缺少 ```python 代码块")
    if has_code_block and not has_enough_code:
        failed_reasons.append("Python 代码不少于 15 行")
    if has_code_block and not has_function_or_class:
        failed_reasons.append("Python 代码应包含至少一个函数或类")
    if has_code_block and not syntax_ok:
        failed_reasons.append(f"Python 语法错误: {syntax_error}")
    if not has_run_instruction:
        failed_reasons.append("缺少运行方式说明")
    if not has_expected_output:
        failed_reasons.append("缺少预期输出说明")
    if not has_error_debugging:
        failed_reasons.append("缺少错误排查说明")
    if not has_extension_tasks:
        failed_reasons.append("缺少拓展任务")
    if not topic_relevant:
        failed_reasons.append("代码内容和主题相关性不足")

    return {
        "passed": not failed_reasons,
        "failed_reasons": failed_reasons,
        "missing_sections": missing_sections,
        "has_code_block": has_code_block,
        "has_enough_code": has_enough_code,
        "has_function_or_class": has_function_or_class,
        "syntax_ok": syntax_ok,
        "syntax_error": syntax_error,
        "has_run_instruction": has_run_instruction,
        "has_expected_output": has_expected_output,
        "has_error_debugging": has_error_debugging,
        "has_extension_tasks": has_extension_tasks,
        "topic_relevant": topic_relevant,
        "code_line_count": len(code_lines),
    }


def _extract_first_python_code(markdown: str) -> str:
    code = _extract_first_python_code_block(markdown)
    return code + "\n" if code else ""


def _code_practice_model_name() -> str:
    configured_model = get_setting("llm.code_practice.model", None)
    if not isinstance(configured_model, str) or not configured_model.strip():
        raise ValueError("llm.code_practice.model must be explicitly configured")
    return configured_model.strip()


def _code_practice_temperature() -> float:
    configured_temperature = get_setting("llm.code_practice.temperature", None)
    if isinstance(configured_temperature, bool) or not isinstance(
        configured_temperature, (int, float)
    ):
        raise ValueError("llm.code_practice.temperature must be explicitly configured")
    temperature = float(configured_temperature)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("llm.code_practice.temperature must be between 0 and 2")
    return temperature


def _code_practice_max_generation_rounds() -> int:
    configured_rounds = get_setting("llm.code_practice.max_generation_rounds", None)
    if isinstance(configured_rounds, bool) or not isinstance(configured_rounds, int):
        raise ValueError(
            "llm.code_practice.max_generation_rounds must be explicitly configured"
        )
    if configured_rounds < 1:
        raise ValueError("llm.code_practice.max_generation_rounds must be at least one")
    return configured_rounds


def _planner_prompt(state: LearningState, query: str, context: list[dict]) -> str:
    return (
        "请根据用户问题、学习目标、关键词和检索资料，规划一份代码实操资源蓝图。\n\n"
        f"## 用户问题\n{query}\n\n"
        f"## learning_goal\n{state.get('learning_goal', '') or '未提供'}\n\n"
        f"## keypoints / expanded_keypoints\n{_format_keypoints(state)}\n\n"
        f"## context\n{_format_context(context)}\n\n"
        "## 输出要求\n"
        "只输出蓝图，不要生成完整正文。蓝图必须包含以下部分：\n"
        "- 实操主题\n"
        "- 案例场景\n"
        "- 前置知识\n"
        "- 核心任务\n"
        "- 代码模块拆分\n"
        "- 输入输出设计\n"
        "- 常见错误\n"
        "- 拓展任务\n"
        "不得编造教材页码、文件名或不存在的来源。"
    )


def _agent_prompt(state: LearningState, outline: str) -> str:
    return (
        "请根据代码实操蓝图生成一份 Markdown 代码实操文档。\n\n"
        f"## 用户问题\n{_last_human_query(state)}\n\n"
        f"## 代码实操蓝图\n{outline}\n\n"
        f"## 检索资料\n{_format_context(state.get('context', []))}\n\n"
        f"## 修订意见\n{state.get('code_practice_revision_notes', '') or 'None'}\n\n"
        "## 必须满足的 Markdown 结构\n"
        "# 标题\n"
        "## 一、实操目标\n"
        "## 二、前置知识\n"
        "## 三、案例场景\n"
        "## 四、完整代码\n"
        "## 五、代码逐段讲解\n"
        "## 六、运行方式\n"
        "## 七、预期输出\n"
        "## 八、常见错误与排查\n"
        "## 九、拓展任务\n"
        "## 十、自测问题\n\n"
        "完整代码必须放在 fenced code block 中，格式必须是：\n"
        "```python\n"
        "# code here\n"
        "```\n\n"
        "代码应尽量可直接运行，优先给出单文件 Python 示例。"
        "如果资料不足，请在文末说明资料依据不足。"
        "不要编造教材页码、文件名或不存在的来源。"
    )


@traced_node
async def code_practice_planner(state: LearningState) -> dict:
    query = _last_human_query(state)
    context = state.get("context", [])
    emit_a3_trace(
        logger,
        "code_practice_planner",
        {
            "context_count": len(context),
            "dual_source_mode": bool(state.get("dual_source_mode")),
            "evidence_judge_state": state.get("evidence_judge_state", ""),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    outline = await invoke_plain_llm_fail_fast(
        node_name="code_practice_planner",
        llm_node="code_practice",
        messages=[
            SystemMessage(
                content="You are a university course code-practice planner. Return a concrete blueprint only."
            ),
            HumanMessage(content=_planner_prompt(state, query, context)),
        ],
        state=state,
        temperature=_code_practice_temperature(),
    )
    if not outline.strip():
        raise ValueError("code_practice_planner produced empty outline")
    return {
        "code_practice_outline": outline,
        "code_practice_markdown": "",
        "code_practice_artifact": {},
        "code_practice_review_verdict": "",
        "code_practice_review_reason": "",
        "code_practice_revision_notes": "",
        "code_practice_round": 0,
    }


@traced_node
async def code_practice_agent(state: LearningState) -> dict:
    outline = state.get("code_practice_outline", "")
    if not outline.strip():
        raise CodePracticeGenerationError("code_practice outline is empty")

    round_no = int(state.get("code_practice_round", 0) or 0) + 1
    if (
        state.get("degraded_generation") is True
        and state.get("evidence_judge_state") == "insufficient"
    ):
        raise CodePracticeGenerationError(
            "code_practice generation blocked because evidence is insufficient"
        )
    markdown = await invoke_plain_llm_fail_fast(
        node_name="code_practice_agent",
        llm_node="code_practice",
        messages=[
            SystemMessage(
                content="You are a code-practice case writer. Return Markdown only."
            ),
            HumanMessage(content=_agent_prompt(state, outline)),
        ],
        state=state,
        temperature=_code_practice_temperature(),
    )
    if not markdown.strip():
        raise CodePracticeGenerationError("code_practice_agent produced empty markdown")

    emit_a3_trace(
        logger,
        "code_practice_agent",
        {
            "markdown_chars": len(markdown),
            "round": round_no,
            "context_count": len(state.get("context", [])),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "code_practice_markdown": markdown.strip(),
        "code_practice_artifact": {"title": _extract_markdown_title(markdown)},
        "code_practice_round": round_no,
        "code_practice_review_verdict": "",
        "code_practice_review_reason": "",
    }


@traced_node
async def code_practice_reviewer(state: LearningState) -> dict:
    markdown = state.get("code_practice_markdown", "")
    local_check = _local_check_code_practice(markdown, state)

    def trace_payload(verdict: str, reason: str) -> dict:
        return {
            "local_check_passed": bool(local_check.get("passed")),
            "missing_sections": local_check.get("missing_sections", []),
            "has_code_block": bool(local_check.get("has_code_block")),
            "syntax_ok": bool(local_check.get("syntax_ok")),
            "has_run_instruction": bool(local_check.get("has_run_instruction")),
            "has_expected_output": bool(local_check.get("has_expected_output")),
            "has_error_debugging": bool(local_check.get("has_error_debugging")),
            "has_extension_tasks": bool(local_check.get("has_extension_tasks")),
            "topic_relevant": bool(local_check.get("topic_relevant")),
            "code_line_count": int(local_check.get("code_line_count") or 0),
            "verdict": verdict,
            "reason": reason,
            "markdown_chars": len(markdown),
        }

    if not local_check["passed"]:
        reason = "; ".join(str(item) for item in local_check["failed_reasons"])
        emit_a3_trace(
            logger,
            "code_practice_reviewer",
            trace_payload("revise", reason),
            state=state,
            env_flag="LOG_GENERATION_SUMMARY",
        )
        return {
            "code_practice_review_verdict": "revise",
            "code_practice_review_reason": reason,
            "code_practice_revision_notes": reason,
            "code_practice_local_check": local_check,
        }

    model_name = _code_practice_model_name()
    with traced_llm_call(
        model_name=model_name, node_name="code_practice_reviewer", temperature=0.0
    ):
        structured_result = await invoke_structured_llm(
            node_name="code_practice_reviewer",
            llm_node="code_practice",
            schema=CodePracticeReviewVerdict,
            messages=[
                SystemMessage(
                    content="You are a strict code-practice teaching-quality reviewer. Return only JSON."
                ),
                HumanMessage(
                    content=(
                        "Review the teaching quality of this Markdown code-practice case.\n"
                        "The deterministic local check has already verified structure, Python syntax, run steps, expected output, troubleshooting, extension tasks, and topic relevance.\n"
                        "Return approve only if the explanation is clear, the task is teachable, and the example is useful for the learner. "
                        "Return revise when it can be corrected, or reject when it is fundamentally unusable.\n\n"
                        'JSON shape: {"verdict": "approve", "revise", or "reject", "reason": "..."}\n\n'
                        f"## User question\n{_last_human_query(state)}\n\n"
                        f"## Outline\n{state.get('code_practice_outline', '')}\n\n"
                        f"## Markdown\n{markdown}"
                    )
                ),
            ],
            output_mode=get_llm_output_mode("code_practice_reviewer"),
            business_validator=validate_code_practice_verdict,
            state=state,
            max_raw_chars=get_max_raw_chars("code_practice_reviewer"),
        )
    if not structured_result.success:
        raise StructuredOutputError(structured_result)
    result = structured_result.parsed
    if not isinstance(result, CodePracticeReviewVerdict):
        raise CodePracticeReviewError(
            "code_practice_reviewer parsed result is not CodePracticeReviewVerdict"
        )
    verdict = result.verdict
    reason = result.reason.strip()
    emit_a3_trace(
        logger,
        "code_practice_reviewer",
        trace_payload(verdict, reason),
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "code_practice_review_verdict": verdict,
        "code_practice_review_reason": reason,
        "code_practice_revision_notes": "" if verdict == "approve" else reason,
        "code_practice_local_check": local_check,
    }


@traced_node
async def code_practice_rewrite(state: LearningState) -> dict:
    reason = state.get("code_practice_review_reason", "")
    if not reason.strip():
        raise ValueError("code_practice rewrite requested without review reason")
    return {
        "code_practice_revision_notes": f"Revise the code-practice Markdown according to reviewer feedback:\n{reason}",
        "code_practice_outline": state.get("code_practice_outline", ""),
    }


@traced_node
async def code_practice_output(state: LearningState) -> dict:
    markdown = state.get("code_practice_markdown", "")
    if not markdown.strip():
        raise CodePracticeApprovalError("code_practice markdown is empty")

    review_verdict = str(state.get("code_practice_review_verdict") or "").strip()
    review_reason = str(state.get("code_practice_review_reason") or "").strip()
    if review_verdict != "approve":
        detail = f": {review_reason}" if review_reason else ""
        raise CodePracticeApprovalError(
            f"code_practice output requires an approve verdict{detail}"
        )
    local_check = _local_check_code_practice(markdown, state)
    if not local_check["passed"]:
        raise CodePracticeApprovalError(
            "code_practice output failed local quality check: "
            + "; ".join(str(item) for item in local_check["failed_reasons"])
        )
    title = _extract_markdown_title(markdown)
    python_code = _extract_first_python_code(markdown)
    if not python_code.strip():
        raise CodePracticeApprovalError(
            "code_practice output requires a real Python code block"
        )
    document_artifact = create_document_artifact(
        markdown_text=markdown,
        title=title,
        artifact_kind="code_practice",
    )
    python_filename = Path(document_artifact["filename"]).with_suffix(".py").name
    artifact_dir = get_code_practice_artifact_dir() / str(
        document_artifact["artifact_id"]
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    python_path = artifact_dir / python_filename
    python_path.write_text(python_code, encoding="utf-8")
    python_url = (
        f"/artifacts/code-practice/{document_artifact['artifact_id']}/{python_filename}"
    )

    artifact = {
        **document_artifact,
        "title": title,
        "python_filename": python_filename,
        "python_url": python_url,
        "markdown": markdown,
        "review_reason": review_reason,
    }
    emit_a3_trace(
        logger,
        "code_practice_output",
        {
            "title": title,
            "markdown_chars": len(markdown),
            "review_reason": review_reason,
            "markdown_url": artifact.get("markdown_url", ""),
            "docx_url": artifact.get("docx_url", ""),
            "python_url": artifact.get("python_url", ""),
            "has_python_artifact": bool(python_url),
            "emits_ai_message": True,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    return {
        "code_practice_artifact": artifact,
        "code_practice_markdown": markdown,
        "messages": [AIMessage(content=markdown)],
    }


def should_rewrite_code_practice(state: LearningState) -> str:
    verdict = str(state.get("code_practice_review_verdict") or "").strip()
    if verdict == "approve":
        return "output"
    if verdict not in {"revise", "reject"}:
        raise CodePracticeApprovalError(
            "code_practice routing requires approve, revise, or reject"
        )
    current_round = int(state.get("code_practice_round", 0) or 0)
    if current_round < _code_practice_max_generation_rounds():
        return "rewrite"
    raise CodePracticeApprovalError(
        "code_practice remained unapproved after the maximum rewrite rounds"
    )
