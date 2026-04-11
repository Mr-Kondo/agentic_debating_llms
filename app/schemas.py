"""Pydantic schemas for structured outputs and discussion artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FacilitatorDecision(BaseModel):
    """Structured decision emitted by the facilitator model."""

    action: Literal["speak_a", "speak_b", "search", "finish"]
    reason: str = Field(min_length=1, max_length=500)
    focus_instruction: str | None = Field(default=None, max_length=500)
    search_query: str | None = Field(default=None, max_length=200)
    terminate_reason: str | None = Field(default=None, max_length=300)


class ContinuationDecision(BaseModel):
    """Structured decision emitted by the continuation facilitator."""

    action: Literal["continue_a", "continue_b", "search", "conclude"]
    reason: str = Field(min_length=1, max_length=500)
    focus_instruction: str | None = Field(default=None, max_length=500)
    search_query: str | None = Field(default=None, max_length=200)
    conclude_reason: str | None = Field(default=None, max_length=300)


class DebaterResponse(BaseModel):
    """Structured response emitted by a debater model."""

    speaker: str  # expected "A" or "B"; enforced in debater node
    claim: str = Field(min_length=1, max_length=1200)
    stance_summary: str = Field(min_length=1, max_length=250)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_search: bool = False
    search_query: str | None = Field(default=None, max_length=200)
    search_reason: str | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def validate_search_request(self) -> "DebaterResponse":
        if self.needs_search and not (self.search_query and self.search_query.strip()):
            raise ValueError("search_query is required when needs_search is true")
        return self


class ValidatorFeedback(BaseModel):
    """Structured quality feedback emitted by validator model."""

    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: str = Field(min_length=1, max_length=1200)
    improvement: str = Field(min_length=1, max_length=1200)
    needs_search: bool = False
    search_query: str | None = Field(default=None, max_length=200)
    search_reason: str | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def validate_search_request(self) -> "ValidatorFeedback":
        if self.needs_search and not (self.search_query and self.search_query.strip()):
            raise ValueError("search_query is required when needs_search is true")
        return self


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
