"""Data structures that can be reused across modules."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    query: str = Field(max_length=4096)
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    """Resume a graph interrupted by Human-in-the-loop."""

    thread_id: str
    edited_plan: str = Field(default="", max_length=16384)
    feedback: str | None = Field(default=None, max_length=4096)
    memory_use_choice: Literal["use", "ignore"] | None = None
