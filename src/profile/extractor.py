"""
Profile extractor — uses LLM structured output to extract user traits from conversations.

Design:
- One LLM call per conversation turn (batch mode supported)
- Structured output via invoke_structured_llm / deepseek_tool_call_strict
- Only extracts what is clearly evidenced
- Returns ExtractedProfileInfo (all fields optional)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.structured_output import (
    StructuredOutputError,
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.profile.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt
from src.profile.schema import ExtractedProfileInfo, ExtractedProfileInfoStrict, UserProfile, profile_to_summary

logger = logging.getLogger(__name__)

# Maximum conversation length (chars) to send to the LLM for extraction.
# Longer conversations are truncated from the beginning (most recent is most relevant).
_MAX_CONVERSATION_CHARS = 4000


def _truncate_conversation(text: str, max_chars: int = _MAX_CONVERSATION_CHARS) -> str:
    """Keep the tail of the conversation; recent messages are most relevant."""
    if len(text) <= max_chars:
        return text
    return "…(早期对话已截断)…\n" + text[-max_chars:]


def _format_conversation(
    user_message: str,
    assistant_response: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Format a conversation for the extraction prompt."""
    lines: list[str] = []
    if history:
        for turn in history[-10:]:  # Last 10 turns max
            role = "用户" if turn.get("role") == "user" else "AI"
            content = turn.get("content", "")[:500]
            lines.append(f"{role}: {content}")
    lines.append(f"用户: {user_message[:1000]}")
    lines.append(f"AI: {assistant_response[:1000]}")
    return "\n\n".join(lines)


class ProfileExtractor:
    """Extract user profile information from conversation turns using LLM.

    Usage::

        extractor = ProfileExtractor(llm)
        info = await extractor.extract(
            user_message="什么是闭包？",
            assistant_response="闭包是...（详细解释）...",
            existing_profile=current_profile,
        )
    """

    def __init__(self, llm=None):
        """Initialize the extractor.

        ``llm`` is accepted for compatibility with older callers, but profile
        extraction now resolves provider/model/schema through invoke_structured_llm.
        """
        self._base_llm = llm

    async def extract(
        self,
        user_message: str,
        assistant_response: str,
        existing_profile: UserProfile | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ExtractedProfileInfo:
        """Extract profile signals from a single conversation turn.

        Args:
            user_message: The user's latest message.
            assistant_response: The assistant's latest response.
            existing_profile: Current user profile for context (incremental).
            history: Optional recent conversation history for context.

        Returns:
            ExtractedProfileInfo with any newly observed signals.
            Returns an empty ExtractedProfileInfo if nothing was observed.
        """
        conversation_text = _format_conversation(user_message, assistant_response, history)
        conversation_text = _truncate_conversation(conversation_text)

        existing_summary = ""
        if existing_profile is not None:
            existing_summary = profile_to_summary(existing_profile)

        user_prompt = build_extraction_prompt(conversation_text, existing_summary)

        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        try:
            structured_result = await invoke_structured_llm(
                node_name="profile_extractor",
                llm_node="profile_extractor",
                schema=ExtractedProfileInfoStrict,
                messages=messages,
                output_mode=get_llm_output_mode("profile_extractor"),
                fallback_modes=[],
                state={},
                max_raw_chars=get_max_raw_chars("profile_extractor"),
            )
        except StructuredOutputError as exc:
            logger.warning("Profile extraction structured output failed: %s", exc)
            return ExtractedProfileInfo()
        except Exception as exc:
            logger.warning("Profile extraction LLM call failed: %s", exc)
            return ExtractedProfileInfo()

        if not isinstance(structured_result.parsed, ExtractedProfileInfoStrict):
            logger.warning(
                "Profile extraction returned unexpected parsed type: %s",
                type(structured_result.parsed).__name__,
            )
            return ExtractedProfileInfo()

        result = self._strict_to_extracted(structured_result.parsed)
        # Validate and filter out nonsense.
        result = self._sanitize(result)
        return result

    async def extract_batch(
        self,
        turns: list[dict[str, str]],
        existing_profile: UserProfile | None = None,
    ) -> list[ExtractedProfileInfo]:
        """Extract profile signals from multiple conversation turns.

        Each turn should have: {"user": "...", "assistant": "..."}
        """
        results: list[ExtractedProfileInfo] = []
        for i, turn in enumerate(turns):
            # Build history from previous turns
            history = [
                {"role": "user", "content": t["user"]}
                for t in turns[:i]
            ]
            info = await self.extract(
                user_message=turn["user"],
                assistant_response=turn["assistant"],
                existing_profile=existing_profile,
                history=history,
            )
            results.append(info)
        return results

    @staticmethod
    def _sanitize(info: ExtractedProfileInfo) -> ExtractedProfileInfo:
        """Clean up extracted info: clamp scores and remove noise."""
        # Clamp skill scores
        sanitized_skills: dict[str, float] = {}
        for name, level in info.skills_observed.items():
            if not isinstance(name, str) or len(name) > 50:
                continue
            sanitized_skills[name.strip().lower()] = max(0.0, min(1.0, float(level)))

        # Clamp style signals
        sanitized_style: dict[str, float] = {}
        valid_style_keys = {
            "prefer_examples", "prefer_visual", "prefer_step_by_step",
            "prefer_concise", "prefer_theory", "prefer_practice", "prefer_analogy",
        }
        for key, val in info.style_signals.items():
            if key in valid_style_keys:
                sanitized_style[key] = max(0.0, min(1.0, float(val)))

        # Sanitize goals; importance must be 0-1.
        sanitized_goals: list[dict[str, object]] = []
        for g in info.goals_observed:
            goal_text = str(g.get("goal", "")).strip()
            if not goal_text or len(goal_text) > 200:
                continue
            importance = max(0.0, min(1.0, float(g.get("importance", 0.5))))
            sanitized_goals.append({"goal": goal_text, "importance": importance})

        # Truncate observations
        sanitized_obs = [str(o)[:200].strip() for o in info.observations if str(o).strip()]

        return ExtractedProfileInfo(
            skills_observed=sanitized_skills,
            skill_evidence=str(info.skill_evidence)[:500] if info.skill_evidence else "",
            style_signals=sanitized_style,
            style_evidence=str(info.style_evidence)[:500] if info.style_evidence else "",
            goals_observed=sanitized_goals,
            behavior_update={
                k: float(v) for k, v in (info.behavior_update or {}).items()
                if k in {"avg_session_minutes", "quiz_accuracy", "questions_asked"}
            },
            observations=sanitized_obs,
            dislikes_observed=[str(d)[:100].strip() for d in (info.dislikes_observed or []) if str(d).strip()],
        )

    @staticmethod
    def _strict_to_extracted(info: ExtractedProfileInfoStrict) -> ExtractedProfileInfo:
        """Convert strict-tool DTOs back to the public profile extraction shape."""
        skills: dict[str, float] = {}
        for item in info.skills_observed:
            name = item.name.strip()
            if name:
                skills[name] = float(item.level)

        styles: dict[str, float] = {}
        for item in info.style_signals:
            dimension = item.dimension.strip()
            if dimension:
                styles[dimension] = float(item.strength)

        goals = [
            {"goal": item.goal.strip(), "importance": float(item.importance)}
            for item in info.goals_observed
            if item.goal.strip()
        ]

        behavior_update: dict[str, float] = {}
        behavior = info.behavior_update
        if behavior.avg_session_minutes > 0:
            behavior_update["avg_session_minutes"] = float(behavior.avg_session_minutes)
        if behavior.quiz_accuracy > 0:
            behavior_update["quiz_accuracy"] = float(behavior.quiz_accuracy)
        if behavior.questions_asked > 0:
            behavior_update["questions_asked"] = float(behavior.questions_asked)

        return ExtractedProfileInfo(
            skills_observed=skills,
            skill_evidence=info.skill_evidence,
            style_signals=styles,
            style_evidence=info.style_evidence,
            goals_observed=goals,
            behavior_update=behavior_update,
            observations=list(info.observations),
            dislikes_observed=list(info.dislikes_observed),
        )


def extractor_from_env() -> ProfileExtractor:
    """Factory: build a ProfileExtractor from environment / settings.

    Uses the same LLM configuration as the rest of the project.
    """
    return ProfileExtractor()
