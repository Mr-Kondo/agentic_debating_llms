"""Markdown logger for session-by-session debate records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.schemas import ContinuationDecision, DebaterResponse, FacilitatorDecision, SearchResult, ValidatorFeedback
from app.utils.time_utils import now_utc


@dataclass(slots=True)
class MarkdownLogger:
    """Append structured debate events to markdown file."""

    output_dir: Path
    result_dir: Path | None = None

    def create_session_file(self, session_id: str, topic: str) -> Path:
        """Create markdown file and write session header."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"debate_{session_id}.md"
        header = (
            f"# Debate Session {session_id}\n\n"
            f"- Topic: {topic}\n"
            f"- Started At (UTC): {now_utc().isoformat()}\n\n"
            "## Event Log\n\n"
        )
        path.write_text(header, encoding="utf-8")
        return path

    @staticmethod
    def _append(path: Path, body: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(body)

    def append_facilitator_decision(self, path: Path, decision: FacilitatorDecision) -> None:
        """Append facilitator decision event."""
        body = (
            f"### Facilitator Decision ({now_utc().isoformat()})\n"
            f"- action: {decision.action}\n"
            f"- reason: {decision.reason}\n"
            f"- focus_instruction: {decision.focus_instruction or ''}\n"
            f"- search_query: {decision.search_query or ''}\n"
            f"- terminate_reason: {decision.terminate_reason or ''}\n\n"
        )
        self._append(path, body)

    def append_continuation_decision(self, path: Path, decision: ContinuationDecision) -> None:
        """Append continuation facilitator decision event."""
        body = (
            f"### Continuation Decision ({now_utc().isoformat()})\n"
            f"- action: {decision.action}\n"
            f"- reason: {decision.reason}\n"
            f"- focus_instruction: {decision.focus_instruction or ''}\n"
            f"- search_query: {decision.search_query or ''}\n"
            f"- conclude_reason: {decision.conclude_reason or ''}\n\n"
        )
        self._append(path, body)

    def append_debater_utterance(self, path: Path, response: DebaterResponse) -> None:
        """Append debater utterance event."""
        body = (
            f"### Debater {response.speaker} ({now_utc().isoformat()})\n"
            f"- stance_summary: {response.stance_summary}\n"
            f"- confidence: {response.confidence:.2f}\n\n"
            f"> {response.claim}\n\n"
        )
        self._append(path, body)

    def append_search_result(self, path: Path, result: SearchResult) -> None:
        """Append search result event with digest and metadata."""
        body = (
            f"### Search ({now_utc().isoformat()})\n"
            f"- query: {result.query}\n"
            f"- returncode: {result.returncode}\n"
            f"- stderr: {result.stderr[:300]}\n\n"
            f"Digest:\n\n{result.digest}\n\n"
        )
        self._append(path, body)

    def append_validator_feedback(self, path: Path, feedback: ValidatorFeedback) -> None:
        """Append validator feedback event."""
        body = (
            f"### Validator Feedback ({now_utc().isoformat()})\n"
            f"- is_valid: {feedback.is_valid}\n"
            f"- confidence: {feedback.confidence:.2f}\n"
            f"- issues: {feedback.issues}\n"
            f"- improvement: {feedback.improvement}\n\n"
        )
        self._append(path, body)

    def append_final_summary(self, path: Path, summary: str) -> None:
        """Append final summary and session completion timestamp."""
        body = f"## Final Summary ({now_utc().isoformat()})\n\n{summary}\n\n- Completed At (UTC): {now_utc().isoformat()}\n"
        self._append(path, body)

    def write_result_snapshot(
        self,
        *,
        session_id: str,
        topic: str,
        final_summary: str,
        input_sources: list[str],
        validation_highlights: list[str] | None = None,
    ) -> Path:
        """Write a user-facing final result markdown under out directory."""
        target_dir = self.result_dir or self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"result_{session_id}.md"
        sources = "\n".join(f"- {name}" for name in input_sources) if input_sources else "- (none)"
        highlights = "\n".join(f"- {line}" for line in validation_highlights) if validation_highlights else "- (none)"
        content = (
            f"# Debate Result {session_id}\n\n"
            f"- Topic: {topic}\n"
            f"- Generated At (UTC): {now_utc().isoformat()}\n\n"
            "## Input Sources\n\n"
            f"{sources}\n\n"
            "## Validator Highlights\n\n"
            f"{highlights}\n\n"
            "## Final Summary\n\n"
            f"{final_summary}\n"
        )
        path.write_text(content, encoding="utf-8")
        return path
