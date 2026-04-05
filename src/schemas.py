"""Data structures that can be reused across modules."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    query: str = Field(max_length=4096)
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    """Resume a graph interrupted by Human-in-the-loop."""

    thread_id: str
    edited_plan: str = Field(max_length=16384)
