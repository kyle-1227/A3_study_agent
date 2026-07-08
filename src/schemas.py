"""Data structures that can be reused across modules."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    query: str = Field(max_length=4096)
    thread_id: str | None = None
    user_id: str | None = None


class ResumeRequest(BaseModel):
    """Resume a graph interrupted by Human-in-the-loop."""

    thread_id: str
    edited_plan: str = Field(default="", max_length=16384)
    feedback: str | None = Field(default=None, max_length=4096)
    memory_use_choice: Literal["use", "ignore"] | None = None


class StopRequest(BaseModel):
    """Request a safe stop at the next LangGraph node boundary."""

    reason: str = Field(default="user_stop", max_length=512)


class ThreadStatusResponse(BaseModel):
    """Run-control status for a LangGraph thread checkpoint."""

    thread_id: str
    schema_version: Literal["run_control_v1", "legacy"]
    run_status: str
    resume_available: bool
    pending_interrupt_type: str = ""
    current_node: str = ""
    last_completed_node: str = ""
    stopped_at: str = ""
    stop_reason: str = ""
    context_usage: dict[str, Any] = Field(default_factory=dict)
    context_usage_history: list[dict[str, Any]] = Field(default_factory=list)
    last_llm_input_manifest: dict[str, Any] = Field(default_factory=dict)
    llm_input_manifest_count: int = 0
    background_context_window: dict[str, Any] = Field(default_factory=dict)
    request_context_window: dict[str, Any] = Field(default_factory=dict)
    thread_context_window: dict[str, Any] = Field(default_factory=dict)
    missing_run_control_fields: list[str] = Field(default_factory=list)
    message: str = ""


class OnboardRequest(BaseModel):
    """Onboarding wizard data submitted on first login.

    All values are explicit self-reports from the user, so they
    carry high confidence (0.9) when stored in the profile.
    """

    user_id: str
    nickname: str = Field(default="")
    subjects: list[str] = Field(default_factory=list)
    skill_levels: dict[str, float] = Field(
        default_factory=dict
    )  # subject → 0.25|0.5|0.75
    goals: list[str] = Field(default_factory=list)
    learning_style: dict[str, float] = Field(default_factory=dict)  # dim → 0.2|0.5|0.8
    grade: str | None = None
    dislikes: list[str] | None = None


class ProfileResponse(BaseModel):
    """Profile data returned to the frontend."""

    user_id: str
    has_profile: bool
    summary: str | None = None
