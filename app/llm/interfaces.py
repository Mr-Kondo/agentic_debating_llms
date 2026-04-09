"""Interfaces to keep LLM-dependent logic replaceable."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from app.state import DiscussionState

TModel = TypeVar("TModel", bound=BaseModel)


class StructuredLLM(Protocol):
    """Protocol for requesting structured outputs from an LLM backend."""

    def generate_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[TModel],
        keep_alive: str | None = None,
    ) -> TModel:
        """Generate a structured response validated against schema_model."""


class TextLLM(Protocol):
    """Protocol for plain text generation."""

    def generate_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        keep_alive: str | None = None,
    ) -> str:
        """Generate plain text output."""


class SummaryProvider(Protocol):
    """Protocol for state summarization, swappable with DSPy in the future."""

    def summarize(self, state: DiscussionState) -> str:
        """Produce a compact summary from discussion state."""


class SearchDigestProvider(Protocol):
    """Protocol for converting raw search output into compact digest."""

    def digest(self, stdout: str, stderr: str, returncode: int) -> str:
        """Create bounded digest text for downstream context."""
