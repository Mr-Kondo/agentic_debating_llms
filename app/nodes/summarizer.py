"""State summarization node and default summarizer implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.llm.interfaces import SummaryProvider
from app.schemas import DiscussionTurn
from app.state import DiscussionState, get_recent_turns, latest_search_digest


@dataclass(slots=True)
class RuleBasedSummarizer(SummaryProvider):
    """Compact summarizer that avoids sending full transcript every turn."""

    recent_turns: int = 6
    max_summary_chars: int = 1200

    def _turn_to_line(self, turn: DiscussionTurn | dict[str, Any]) -> str:
        if isinstance(turn, dict):
            role = str(turn.get("role", "unknown"))
            content_raw = str(turn.get("content", ""))
        else:
            role = turn.role
            content_raw = turn.content

        content = " ".join(content_raw.strip().split())
        if len(content) > 180:
            content = content[:177] + "..."
        return f"[{role}] {content}"

    @staticmethod
    def _latest_search_digest_compat(state: DiscussionState) -> str:
        digest = latest_search_digest(state)
        if digest:
            return digest

        results = state.get("search_results", [])
        if not results:
            return ""
        latest = results[-1]
        if isinstance(latest, dict):
            return str(latest.get("digest", ""))
        return ""

    def summarize(self, state: DiscussionState) -> str:
        recent = get_recent_turns(state, self.recent_turns)
        lines = [self._turn_to_line(turn) for turn in recent]
        search_digest = self._latest_search_digest_compat(state)
        base = " | ".join(lines)
        if search_digest:
            base = f"{base} || search: {search_digest}"

        merged = f"{state['compact_summary']} || {base}" if state["compact_summary"] else base
        merged_clean = " ".join(merged.split())
        if len(merged_clean) > self.max_summary_chars:
            return merged_clean[-self.max_summary_chars :]
        return merged_clean


def summarizer_node(state: DiscussionState, summarizer: SummaryProvider) -> dict:
    """Update compact summary from bounded context."""
    summary = summarizer.summarize(state)
    return {"compact_summary": summary}
