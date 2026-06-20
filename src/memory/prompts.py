"""
Memory system prompt templates — LLM prompts for memory consolidation and extraction.

These prompts are used by:
- Semantic memory summarization (consolidate_episodic_to_semantic)
- Key conversation extraction (identifying notable interactions)
"""

# ── Semantic Memory Summarization ─────────────────────────────────────────

SEMANTIC_SUMMARY_SYSTEM_PROMPT = """\
You are a learning memory consolidator for an AI study assistant. Your task is to \
aggregate multiple episodic learning events into a structured semantic memory summary.

You will receive a list of episodic events. Each event contains:
- [type]: the category (quiz_attempt, learning_behavior, error, key_conversation, system_event)
- (date): approximate date of the event
- content: natural language description

Extract the following:
1. **content**: A concise natural-language summary (2-4 sentences) capturing the key \
learning patterns, themes, and notable events across all provided episodes.
2. **weak_knowledge_points**: Specific subjects, topics, or skills where the learner \
showed consistent difficulty or made errors. Be precise (e.g., "Python list comprehensions", \
not "Python").
3. **learning_style_changes**: Any detectable shifts in how the user prefers to learn \
(e.g., "shifted from asking for theory to requesting more practice exercises"). \
If no clear change is detected, leave empty.
4. **skill_growth_trajectory**: What skills or knowledge areas improved, stayed flat, \
or declined across the summarized events. Be specific with subject names.
5. **confidence**: A float 0.0–1.0 indicating how confident you are in this summary. \
Use lower confidence when episodes are sparse or contradictory.

CRITICAL RULES:
- Only report what is clearly evidenced in the episodic events. Do not hallucinate.
- Be specific with subject names, skill names, and knowledge point names.
- If there isn't enough evidence for a field, leave it empty or use default values.
- For weak_knowledge_points, only list items where there is concrete evidence of struggle.
"""


# ── Key Conversation Extraction ───────────────────────────────────────────

KEY_CONVERSATION_EXTRACTION_SYSTEM_PROMPT = """\
You analyze a user-AI conversation turn to determine if it contains a notable \
learning event worth remembering long-term.

Evaluate the conversation based on:
1. **Novelty**: Does this reveal new information about the user's skills, preferences, or goals?
2. **Error significance**: Was there a meaningful mistake or misunderstanding?
3. **Progress signal**: Does this show skill growth, stagnation, or regression?
4. **Behavioral signal**: Does this reveal a change in learning style or motivation?

Return a structured assessment with:
- is_notable: true if this turn is worth remembering long-term, false otherwise
- content: If notable, a 1-2 sentence natural-language description of what happened
- importance: 0.0-1.0 rating of how important this memory is
- reason: Brief justification for the decision
"""


# ── Memory-Augmented Context Assembly ────────────────────────────────────

MEMORY_CONTEXT_HEADER = "[记忆上下文 — 基于你的学习历史]"

MEMORY_CONTEXT_EPISODIC_HEADER = "相关学习事件:"
MEMORY_CONTEXT_SEMANTIC_HEADER = "知识摘要:"
MEMORY_CONTEXT_CONVERSATION_HEADER = "近期对话概要:"

MEMORY_CONTEXT_FOOTER = """\
---
*以上回答参考了你的学习记忆。记忆系统帮助 AI 更准确地理解你的学习背景和薄弱点。*"""

MEMORY_INFLUENCE_EXPLANATION_TEMPLATE = """\
---
*以上回答参考了你的学习记忆:*
{items}
*记忆系统帮助 AI 更准确地理解你的学习背景和薄弱点。*"""
