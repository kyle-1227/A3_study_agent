"""
Prompt templates for profile extraction and summarization.

These prompts are designed for structured output (JSON mode).
They instruct the LLM to extract ONLY what is clearly evidenced
in the conversation — no hallucinated traits.
"""

from __future__ import annotations

# ── Profile extraction prompt ──────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """你是一个学习画像分析系统。你的任务是从用户与AI助教的对话中提取用户特征。

## 提取原则

1. **只提取有明确证据的信息** — 不要猜测或编造
2. **使用连续分数 (0.0–1.0)** — 不要用"初级/中级/高级"这种离散标签
3. **每条提取都要有证据** — 在 evidence 字段中引用对话原文或总结
4. **置信度反映证据强度** — 明确陈述 > 暗示 > 推测
5. **不要提取无意义信息** — 如果本轮对话没有新的用户特征，返回空值

## 提取维度

### 技能水平 (skills_observed)
- 从用户的问题深度、术语使用、代码质量等方面判断
- 0.0 = 完全不了解, 0.3 = 入门, 0.5 = 中等, 0.8 = 熟练, 1.0 = 专家
- 例如：用户问"Python的list和tuple有什么区别" → python: 0.25
- 例如：用户讨论GIL原理和C扩展优化 → python: 0.85
- 例如：用户说"我刚学Python两周" → python: 0.1 (confidence: 0.9)

### 学习偏好 (style_signals)
- prefer_examples: 用户是否要求/偏好代码案例或具体例子 (0→1)
- prefer_visual: 用户是否偏好图表、可视化
- prefer_step_by_step: 用户是否要求或表现出对分步讲解的偏好
- prefer_concise: 用户是否偏好简短直接的回答
- prefer_theory: 用户是否追问原理和理论
- prefer_practice: 用户是否要求练习题或实操
- prefer_analogy: 用户是否喜欢比喻和类比

### 学习目标 (goals_observed)
- 用户明确或隐含的学习目标
- 例如："我准备考研408" → goal: "准备408考研", importance: 0.9
- 例如："我想学Python做数据分析" → goal: "Python数据分析", importance: 0.7

### 行为特征 (behavior_update)
- 可量化的行为指标（如有）

### Agent观察 (observations)
- 用自然语言记录的重要发现
- 例如："用户在算法题上需要较多提示"
- 例如："用户对递归概念理解较好"
- 每条不超过30字

### 不喜欢 (dislikes_observed)
- 用户明确表示不喜欢或回避的主题/方式

## 输出格式

只返回 JSON，不要有其他文字。
如果本轮没有新发现，返回空对象 {}。
"""

EXTRACTION_USER_TEMPLATE = """## 对话历史
{conversation_text}

## 当前已有画像（供参考，请在此基础上增量更新）
{existing_summary}

请从以上对话中提取本轮新观察到的用户特征。只提取本轮新出现的、有明确证据的信息。"""


# ── Profile summarization prompt ───────────────────────────────────────────

SUMMARIZATION_SYSTEM_PROMPT = """你是一个学习画像摘要系统。请将用户画像数据压缩为简洁的自然语言摘要，用于注入到AI助教的系统提示中。

## 摘要要求

1. **简洁** — 控制在200字以内
2. **可操作** — 提供具体的教学策略建议
3. **分级** — 最重要的信息放在前面
4. **不要重复** — 相似信息合并
5. **策略导向** — 不只是描述用户，还要建议AI应该如何调整教学

## 输出格式

直接输出自然语言摘要，不要JSON。
"""

SUMMARIZATION_USER_TEMPLATE = """用户画像数据：
{profile_json}

请生成教学策略导向的用户画像摘要。"""


# ── Profile update / merge prompt ──────────────────────────────────────────

UPDATE_SYSTEM_PROMPT = """你是一个学习画像更新系统。你需要根据新的观察来更新用户的技能评分和学习偏好。

## 更新原则

1. **贝叶斯式更新** — 新证据应该逐步调整分数，而不是完全替换
2. **置信度反映证据量** — 证据越多，置信度越高
3. **衰减旧信息** — 如果新证据与旧评估矛盾，适度降低旧评估的置信度
4. **保持连续性** — 一次对话不应该让技能分数变化超过0.3

## 输出格式

返回 JSON，包含需要更新的字段和新的值。
只返回确实需要更新的字段。
"""


def build_extraction_prompt(conversation_text: str, existing_summary: str = "") -> str:
    """Build the user message for profile extraction."""
    return EXTRACTION_USER_TEMPLATE.format(
        conversation_text=conversation_text[:4000],  # Truncate for token budget
        existing_summary=existing_summary or "（暂无已有画像）",
    )
