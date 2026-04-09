"""Pydantic schemas for structured outputs and discussion artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class FacilitatorDecision(BaseModel):
    """Structured decision emitted by the facilitator model."""

    action: Literal["speak_a", "speak_b", "search", "finish"]
    reason: str = Field(min_length=1, max_length=500)
    focus_instruction: str | None = Field(default=None, max_length=500)
    search_query: str | None = Field(default=None, max_length=200)
    terminate_reason: str | None = Field(default=None, max_length=300)


class DebaterResponse(BaseModel):
    """Structured response emitted by a debater model."""

    speaker: Literal["A", "B"]
    claim: str = Field(min_length=1, max_length=1200)
    stance_summary: str = Field(min_length=1, max_length=250)
    confidence: float = Field(ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Result of a single search CLI execution."""

    query: str
    stdout: str
    stderr: str
    returncode: int
    digest: str


class DiscussionTurn(BaseModel):
    """Single turn in the discussion transcript."""

    role: str
    content: str
    timestamp: datetime
