"""State definitions for LangGraph orchestration."""

from __future__ import annotations

from typing import Any, TypedDict

from app.schemas import DiscussionTurn, FacilitatorDecision, SearchResult, ValidatorFeedback


class DiscussionState(TypedDict):
    """Global state carried across all nodes."""

    topic: str
    transcript: list[DiscussionTurn]
    search_results: list[SearchResult]
    validation_log: list[ValidatorFeedback]
    compact_summary: str
    turn_count: int
    max_turns: int
    next_action: str
    last_decision: FacilitatorDecision | dict[str, Any]
    final_summary: str | None
    markdown_path: str
    result_markdown_path: str | None
    input_sources: list[str]
    session_id: str
    last_error: str | None
    # Continuation mode fields (active only when continuation_max_turns > 0)
    continuation_mode: bool
    continuation_turn_count: int
    continuation_max_turns: int


def get_recent_turns(state: DiscussionState, count: int) -> list[DiscussionTurn]:
    """Return the most recent transcript turns, bounded by count."""
    if count <= 0:
        return []
    return state["transcript"][-count:]


def latest_search_digest(state: DiscussionState) -> str:
    """Return the latest search digest if present."""
    if not state["search_results"]:
        return ""
    latest = state["search_results"][-1]
    if isinstance(latest, dict):
        return str(latest.get("digest", ""))
    return latest.digest
