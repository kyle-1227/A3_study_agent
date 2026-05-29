"""Supervisor node — LLM-based intent classification, subject detection, and keypoint extraction.

Combines routing and academic keypoint extraction into a single LLM call
to eliminate a redundant API roundtrip on the academic path.
Uses structured output (Pydantic) instead of manual JSON parsing.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.config import get_setting, load_prompt
from src.graph.llm import get_node_llm
from src.graph.state import TutorState
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)


class SupervisorOutput(BaseModel):
    """Structured output for supervisor intent classification."""
    intent: Literal["academic", "planning", "emotional", "unknown"]
    keywords: list[str]
    confidence: float


_VALID_INTENTS = set(get_setting(
    "supervisor.valid_intents",
    ["academic", "planning", "emotional", "unknown"],
))


@traced_node
async def supervisor_node(state: TutorState) -> dict:
    """Classify intent, detect subject, and extract keypoints in one LLM call.

    Uses ``with_structured_output(SupervisorOutput)`` for reliable parsing.

    Returns:
        Dict with ``intent``, ``subject``, and ``keypoints`` for state update.
    """
    llm = get_node_llm("supervisor")
    structured_llm = llm.with_structured_output(SupervisorOutput)

    last_msg = state["messages"][-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    temperature = get_setting("supervisor.temperature", 0.0)
    model_name = get_setting("supervisor.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    with traced_llm_call(
        model_name=model_name,
        node_name="supervisor",
        temperature=temperature,
    ):
        try:
            result = await structured_llm.ainvoke([
                SystemMessage(content=load_prompt("supervisor_system")),
                HumanMessage(content=user_text),
            ])
            intent = result.intent
            subject = "other"
            keypoints = result.keywords
            # Detect subject from structured output context
            if intent == "academic" and keypoints:
                query_lower = user_text.lower()
                math_keywords = {"数学", "函数", "方程", "几何", "代数", "概率", "向量",
                                 "导数", "积分", "椭圆", "双曲线", "抛物线", "三角"}
                chinese_keywords = {"语文", "作文", "文言文", "古诗", "阅读理解", "诗词",
                                    "鉴赏", "修辞", "散文", "小说"}
                if any(kw in query_lower for kw in math_keywords):
                    subject = "math"
                elif any(kw in query_lower for kw in chinese_keywords):
                    subject = "chinese"
        except Exception:
            logger.warning("Supervisor structured output failed, defaulting to academic")
            intent = "academic"
            subject = "other"
            keypoints = []

    requested_resource_type = _detect_requested_resource_type(user_text)
    needs_mindmap = requested_resource_type == "mindmap"

    if intent not in _VALID_INTENTS:
        intent = "academic"

    return {
        "intent": intent,
        "subject": subject,
        "keypoints": keypoints,
        "requested_resource_type": requested_resource_type,
        "needs_mindmap": needs_mindmap,
    }


@traced_node
async def handle_unknown(state: TutorState) -> dict:
    """Handle off-topic queries with a friendly redirect message."""
    return {
        "messages": [AIMessage(
            content=(
                "抱歉，这个问题超出了我的辅导范围。我是你的高校学习助手，"
                "可以帮你探索专业方向、解答课程知识、制定学习路径、生成学习资源，"
                "或者聊聊学习中的困惑。请问有什么需要帮助的吗？"
            ),
        )],
    }


def route_by_intent(state: TutorState) -> str:
    """Conditional edge function: route to the appropriate subgraph."""
    if state.get("needs_mindmap"):
        return "mindmap"
    return state.get("intent", "academic")


_RESOURCE_ACTION_MARKERS = (
    "生成",
    "制作",
    "创建",
    "导出",
    "整理成",
    "转成",
    "输出",
    "来一份",
    "做一个",
    "做一份",
    "画一个",
    "画一份",
    "设计一份",
    "帮我做",
    "帮我生成",
    "给我做",
    "给我生成",
    "generate",
    "create",
    "export",
    "make",
)

_WEAK_REQUEST_MARKERS = ("帮我", "给我", "我要", "我想要", "请")

_EXPLANATION_MARKERS = (
    "是什么",
    "什么是",
    "怎么理解",
    "如何理解",
    "为什么",
    "讲讲",
    "解释",
    "介绍",
    "原理",
    "区别",
    "作用",
    "用途",
)

_RESOURCE_TYPE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mindmap", ("思维导图", "知识图谱", "脑图", "结构图", "mindmap", "mind map", "markmap", "xmind")),
    ("quiz", ("练习题", "分层练习", "题库", "习题", "测验", "测试题", "题目")),
    ("ppt", ("ppt", "幻灯片", "演示文稿", "课件")),
    ("code_case", ("代码案例", "代码实操", "实操案例", "编程案例", "示例代码", "demo")),
    ("project_case", ("项目案例", "实践项目", "项目实战", "课程项目", "实验项目")),
    ("video_script", ("视频脚本", "动画脚本", "讲解视频", "教学视频", "分镜")),
    ("reading", ("拓展阅读", "阅读材料", "参考资料", "文献清单", "资料清单")),
    ("other", ("讲义", "学习资源", "资源清单", "学习材料", "复习资料", "知识卡片")),
)


def _detect_requested_resource_type(text: str) -> str:
    """Deterministically identify explicit resource-generation requests.

    A resource type only counts when the user asks to create/export/produce it.
    Explanation questions such as "思维导图是什么" remain ordinary tutoring.
    """
    lowered = text.lower()
    has_strong_action = any(marker.lower() in lowered for marker in _RESOURCE_ACTION_MARKERS)
    has_weak_request = any(marker.lower() in lowered for marker in _WEAK_REQUEST_MARKERS)
    asks_explanation = any(marker.lower() in lowered for marker in _EXPLANATION_MARKERS)

    if not has_strong_action and (not has_weak_request or asks_explanation):
        return ""

    for resource_type, markers in _RESOURCE_TYPE_MARKERS:
        if any(marker.lower() in lowered for marker in markers):
            return resource_type
    return ""
