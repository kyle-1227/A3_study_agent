"""Collaborative leveled-exercise resource-generation nodes."""

from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_setting, load_prompt
from src.graph.json_output import ainvoke_strict_json
from src.graph.llm import async_invoke_with_fallback, get_fallback_llm, get_node_llm
from src.graph.state import TutorState
from src.observability.a3_trace import emit_a3_trace
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)

REQUIRED_LEVELS = {"基础题", "进阶题", "应用题", "自我检查题"}


class ExerciseItem(BaseModel):
    """A single exercise item with answer and teaching feedback."""

    level: str = Field(description="One of 基础题, 进阶题, 应用题, 自我检查题")
    question: str = Field(default="", description="Exercise question")
    answer: str = Field(default="", description="Concise reference answer")
    explanation: str = Field(default="", description="Step-by-step explanation")
    pitfall: str = Field(default="", description="Common mistake or reminder")
    tags: list[str] = Field(default_factory=list, description="Related knowledge points")


class ExerciseArtifact(BaseModel):
    """Structured exercise resource produced by exercise_agent."""

    title: str
    items: list[ExerciseItem]


class ExerciseReviewVerdict(BaseModel):
    """Structured quality gate output for exercise_reviewer."""

    verdict: Literal["approve", "reject"]
    reason: str


def _last_human_query(state: TutorState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _model_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _format_keypoints(state: TutorState) -> str:
    keypoints = state.get("keypoints", [])
    return "、".join(keypoints) if keypoints else "未提取到明确关键词"


def _subjects_used(context: list[dict]) -> list[str]:
    return sorted({str(item.get("retrieval_subject")) for item in context if item.get("retrieval_subject")})


def _roles_used(context: list[dict]) -> list[str]:
    return sorted({str(item.get("retrieval_role")) for item in context if item.get("retrieval_role")})


def _is_web_evidence(item: dict) -> bool:
    return (
        item.get("source_type") == "web"
        or item.get("type") in {"web_evidence", "web_supplement"}
        or item.get("legacy_type") == "web_supplement"
        or item.get("type_legacy") == "web_supplement"
    )


def _web_evidence_items(context: list[dict]) -> list[dict]:
    return [item for item in context if _is_web_evidence(item)]


def _format_context(context: list[dict]) -> str:
    if not context:
        return (
            "当前课程知识库和外部资料未返回可用依据。可以按高校课程通用知识设计分层练习，"
            "但不得编造教材页码、课程政策或虚假引用来源。"
        )

    parts: list[str] = []
    for idx, item in enumerate(context[:8], 1):
        source = item.get("source") or item.get("title") or item.get("url") or "课程资料"
        content = str(item.get("content") or item.get("snippet") or item.get("text") or "")[:700]
        if content:
            parts.append(f"[{idx}] 来源：{source}\n{content}")
    return "\n\n".join(parts) or "已有资料缺少可读正文，请结合用户请求设计通用课程练习。"


def _render_prompt(prompt_name: str, replacements: dict[str, str]) -> str:
    """Render named placeholders without interpreting JSON braces in prompts."""
    prompt = load_prompt(prompt_name)
    for key, value in replacements.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


def _fallback_outline(query: str, keypoints: list[str], context: list[dict]) -> str:
    topic = "、".join(keypoints[:4]) or query[:40] or "课程知识点"
    return (
        f"主题：{topic}\n"
        "1. 基础题：检查核心概念、定义和基本判断。\n"
        "2. 进阶题：考查概念关系、推理步骤和常见变式。\n"
        "3. 应用题：结合代码、实验、项目或真实场景解决问题。\n"
        "4. 自我检查题：帮助学习者复盘掌握程度、易错点和后续补强方向。\n"
        f"依据摘要：{_format_context(context)[:500]}"
    )


def _normalize_items(items: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for item in items[:24]:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "level": str(item.get("level") or "").strip() or "基础题",
                "question": str(item.get("question") or "").strip(),
                "answer": str(item.get("answer") or "").strip(),
                "explanation": str(item.get("explanation") or "").strip(),
                "pitfall": str(item.get("pitfall") or "").strip(),
                "tags": list(item.get("tags") or []),
            }
        )
    return normalized


def _local_review_failure(items: list[dict], _query: str) -> str:
    if len(items) < 4:
        return f"练习题数量只有 {len(items)} 道，至少需要覆盖四类题目。"

    levels = {str(item.get("level", "")).strip() for item in items}
    missing_levels = REQUIRED_LEVELS - levels
    if missing_levels:
        return f"缺少题型层级：{'、'.join(sorted(missing_levels))}。"

    for idx, item in enumerate(items, 1):
        if not item.get("question"):
            return f"第 {idx} 道题缺少题干。"
        if not item.get("answer"):
            return f"第 {idx} 道题缺少答案。"
        if not item.get("explanation"):
            return f"第 {idx} 道题缺少解析。"
        if not item.get("pitfall"):
            return f"第 {idx} 道题缺少易错提醒。"

    return ""


def _render_exercise_markdown(title: str, items: list[dict], *, review_reason: str = "", quality_warning: bool = False) -> str:
    grouped = {level: [] for level in ["基础题", "进阶题", "应用题", "自我检查题"]}
    for item in items:
        grouped.setdefault(item.get("level", "基础题"), []).append(item)

    lines = [f"## {title}", ""]
    if quality_warning:
        lines.extend([f"> 质量提示：审查智能体认为仍存在风险：{review_reason or '题目质量未完全通过审查'}", ""])

    for level in ["基础题", "进阶题", "应用题", "自我检查题"]:
        lines.append(f"### {level}")
        level_items = grouped.get(level) or []
        if not level_items:
            lines.append("- 暂无该层级题目。")
            lines.append("")
            continue
        for idx, item in enumerate(level_items, 1):
            tags = "、".join(str(tag) for tag in item.get("tags", []) if tag)
            lines.extend(
                [
                    f"{idx}. **题目**：{item.get('question', '')}",
                    f"   - **答案**：{item.get('answer', '')}",
                    f"   - **解析**：{item.get('explanation', '')}",
                    f"   - **易错提醒**：{item.get('pitfall', '')}",
                ]
            )
            if tags:
                lines.append(f"   - **关联知识点**：{tags}")
        lines.append("")
    return "\n".join(lines).strip()


@traced_node
async def exercise_planner(state: TutorState) -> dict:
    """Plan leveled exercises from the user request and retrieval context."""
    query = _last_human_query(state)
    keypoints = state.get("keypoints", [])
    context = state.get("context", [])
    web_evidence = _web_evidence_items(context)
    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    emit_a3_trace(
        logger,
        "exercise_planner",
        {
            "subjects_used": _subjects_used(context),
            "roles_used": _roles_used(context),
            "learning_goal": state.get("learning_goal", ""),
            "primary_subject": state.get("primary_subject", ""),
            "context_count": len(context),
            "context_web_count": len(web_evidence),
            "web_supplement_needed": bool(state.get("web_supplement_decisions")),
            "web_supplement_count": len(web_evidence),
            "web_supplement_provider": state.get("web_supplement_provider", "tavily"),
            "web_supplement_failed": bool(state.get("web_supplement_failed")),
            "web_supplement_failure_reason": state.get("web_supplement_failure_reason", ""),
            "web_supplement_partial_failed": bool(state.get("web_supplement_partial_failed")),
            "web_supplement_status_by_subject": state.get("web_supplement_status_by_subject", {}),
            "web_supplement_success_subjects": state.get("web_supplement_success_subjects", []),
            "web_supplement_failed_subjects": state.get("web_supplement_failed_subjects", []),
            "web_evidence_count": len(web_evidence),
            "web_evidence_provider": "tavily",
            "web_judge_provider": state.get("web_judge_provider", "nvidia_build"),
            "web_judge_model": state.get("web_judge_model", "deepseek-ai/deepseek-v4-flash"),
            "web_judge_failed_subjects": state.get("web_judge_failed_subjects", []),
            "web_judge_rejected_all_subjects": state.get("web_judge_rejected_all_subjects", []),
            "web_evidence_use_cases": sorted({item.get("use_case") for item in web_evidence if item.get("use_case")}),
            "web_evidence_types": sorted({item.get("evidence_type") for item in web_evidence if item.get("evidence_type")}),
            "dual_source_mode": bool(state.get("dual_source_mode")),
            "evidence_judge_state": state.get("evidence_judge_state", ""),
            "search_refinement_needed": bool(state.get("search_refinement_needed")),
            "search_refinement_deferred": bool(state.get("search_refinement_deferred")),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    prompt = _render_prompt(
        "exercise_planner",
        {
            "question": query,
            "keypoints": _format_keypoints(state),
            "context": _format_context(context),
        },
    )

    llm = get_node_llm("exercise")
    fallback = get_fallback_llm(temperature=get_setting("exercise.temperature", 0.2))
    temperature = get_setting("exercise.temperature", 0.2)
    model_name = get_setting("exercise.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

    try:
        with traced_llm_call(model_name=model_name, node_name="exercise_planner", temperature=temperature) as span:
            result = await async_invoke_with_fallback(
                llm,
                [
                    SystemMessage(content="你是高校课程分层练习设计规划智能体，负责生成练习蓝图。"),
                    HumanMessage(content=prompt),
                ],
                fallback=fallback,
                span=span,
            )
        outline = str(getattr(result, "content", result)).strip()
    except Exception:
        logger.warning("Exercise planner failed, using fallback outline", exc_info=True)
        outline = _fallback_outline(query, keypoints, context)

    if not outline:
        outline = _fallback_outline(query, keypoints, context)

    return {
        "exercise_outline": outline,
        "exercise_items": [],
        "exercise_artifact": {},
        "exercise_review_verdict": "",
        "exercise_review_reason": "",
        "exercise_revision_notes": "",
        "exercise_round": 0,
    }


@traced_node
async def exercise_agent(state: TutorState) -> dict:
    """Generate structured leveled exercises from the planner outline."""
    query = _last_human_query(state)
    keypoints = state.get("keypoints", [])
    outline = state.get("exercise_outline", "")
    revision_notes = state.get("exercise_revision_notes", "")
    context = state.get("context", [])
    round_no = int(state.get("exercise_round", 0) or 0) + 1

    if not outline.strip():
        return {
            "error": "练习设计蓝图为空，无法生成有质量保障的分层练习题。",
            "exercise_round": round_no,
        }

    prompt = _render_prompt(
        "exercise_agent",
        {
            "question": query,
            "keypoints": _format_keypoints(state),
            "context": _format_context(context),
            "exercise_outline": outline,
            "revision_notes": revision_notes or "暂无审查修订意见。",
        },
    )

    llm = get_node_llm("exercise")
    temperature = get_setting("exercise.temperature", 0.2)
    model_name = get_setting("exercise.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

    try:
        with traced_llm_call(model_name=model_name, node_name="exercise_agent", temperature=temperature) as span:
            result = await ainvoke_strict_json(
                llm,
                [
                    SystemMessage(content="你是分层练习题生成智能体。只输出一个 JSON 对象，不要输出 Markdown、代码块或解释文本。"),
                    HumanMessage(content=prompt),
                ],
                schema=ExerciseArtifact,
                node_name="exercise_agent",
                span=span,
            )
        title = result.title.strip() or "分层练习题"
        raw_items = [_model_to_dict(item) for item in result.items]
    except Exception as exc:
        logger.exception("exercise_agent structured output failed; fallback disabled")
        raise RuntimeError(f"exercise_agent structured output failed; fallback disabled: {exc}") from exc

    items = _normalize_items(raw_items)
    # TEMP A3_TRACE: remove after multi-subject retrieval validation.
    emit_a3_trace(
        logger,
        "exercise_agent",
        {
            "items_count": len(items),
            "levels": [item.get("level") for item in items],
            "subjects_used": _subjects_used(context),
            "roles_used": _roles_used(context),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )

    return {
        "exercise_items": items,
        "exercise_artifact": {"title": title},
        "exercise_round": round_no,
        "exercise_review_verdict": "",
        "exercise_review_reason": "",
    }


@traced_node
async def exercise_reviewer(state: TutorState) -> dict:
    """Review exercises for completeness, leveling, and academic fit."""
    query = _last_human_query(state)
    outline = state.get("exercise_outline", "")
    items = state.get("exercise_items") or []

    local_failure = _local_review_failure(items, query)
    if local_failure:
        return {
            "exercise_review_verdict": "reject",
            "exercise_review_reason": local_failure,
            "exercise_revision_notes": f"请据此重写：{local_failure}",
        }

    prompt = _render_prompt(
        "exercise_reviewer",
        {
            "question": query,
            "exercise_outline": outline,
            "exercise_items": str(items),
        },
    )

    llm = get_node_llm("exercise", temperature=get_setting("exercise.reviewer_temperature", 0.0))
    structured_llm = llm.with_structured_output(ExerciseReviewVerdict, method="json_mode")
    fallback = get_fallback_llm(temperature=get_setting("exercise.reviewer_temperature", 0.0))
    structured_fallback = fallback.with_structured_output(ExerciseReviewVerdict, method="json_mode")
    model_name = get_setting("exercise.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

    try:
        with traced_llm_call(model_name=model_name, node_name="exercise_reviewer", temperature=0.0) as span:
            result = await async_invoke_with_fallback(
                structured_llm,
                [
                    SystemMessage(content="你是高校课程练习题质量审查智能体，只返回 JSON 审查结论。"),
                    HumanMessage(content=prompt),
                ],
                fallback=structured_fallback,
                span=span,
            )
        verdict = result.verdict
        reason = result.reason.strip()
    except Exception:
        logger.warning("Exercise reviewer failed, approving items that passed local checks", exc_info=True)
        verdict = "approve"
        reason = "已通过本地练习结构质量检查。"

    return {
        "exercise_review_verdict": verdict,
        "exercise_review_reason": reason,
        "exercise_revision_notes": "" if verdict == "approve" else f"请据此重写：{reason}",
    }


@traced_node
async def exercise_rewrite(state: TutorState) -> dict:
    """Prepare reviewer feedback for the next exercise generation attempt."""
    reason = state.get("exercise_review_reason", "")
    outline = state.get("exercise_outline", "")
    notes = (
        f"{reason}\n"
        "重写要求：必须覆盖基础题、进阶题、应用题、自我检查题；每题必须包含答案、解析和易错提醒；"
        "题目必须贴合用户课程主题与知识短板。"
    )
    return {
        "exercise_revision_notes": notes,
        "exercise_outline": outline,
    }


@traced_node
async def exercise_output(state: TutorState) -> dict:
    """Render final exercises as Markdown and store structured artifact metadata."""
    items = state.get("exercise_items") or []
    if not items:
        return {
            "error": "当前知识依据不足，未能生成可用的分层练习题。",
            "messages": [AIMessage(content="当前知识依据不足，暂时无法生成质量可靠的分层练习题。请补充课程主题、章节或材料后重试。")],
        }

    title = str((state.get("exercise_artifact") or {}).get("title") or "分层练习题")
    review_verdict = state.get("exercise_review_verdict", "")
    review_reason = state.get("exercise_review_reason", "")
    quality_warning = review_verdict == "reject"
    content = _render_exercise_markdown(
        title,
        items,
        review_reason=review_reason,
        quality_warning=quality_warning,
    )

    return {
        "exercise_artifact": {
            "title": title,
            "items": items,
            "quality_warning": quality_warning,
            "review_reason": review_reason,
        },
        "messages": [AIMessage(content=content)],
    }


def should_rewrite_exercise(state: TutorState) -> str:
    """Route reviewer output to rewrite or final exercise output."""
    if state.get("exercise_review_verdict") != "reject":
        return "output"
    max_rounds = int(get_setting("exercise.max_generation_rounds", 3) or 3)
    current_round = int(state.get("exercise_round", 0) or 0)
    return "rewrite" if current_round < max_rounds else "output"
