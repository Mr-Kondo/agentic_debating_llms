from __future__ import annotations

from datetime import datetime, timezone

from app.nodes.summarizer import RuleBasedSummarizer, summarizer_node


def _turn(role: str, content: str):
    return {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc),
    }


def test_summarizer_uses_recent_turns_and_search_digest() -> None:
    state = {
        "topic": "topic",
        "transcript": [
            _turn("Debater A", "first"),
            _turn("Debater B", "second"),
            _turn("Debater A", "third"),
        ],
        "search_results": [
            {
                "query": "q",
                "stdout": "raw",
                "stderr": "",
                "returncode": 0,
                "digest": "fact digest",
            }
        ],
        "compact_summary": "existing",
        "turn_count": 3,
        "max_turns": 8,
        "next_action": "speak_a",
        "last_decision": {"action": "speak_a", "reason": "x"},
        "final_summary": None,
        "markdown_path": "x.md",
        "session_id": "sid",
        "last_error": None,
    }

    summarizer = RuleBasedSummarizer(recent_turns=2)
    updated = summarizer_node(state, summarizer)

    assert "second" in updated["compact_summary"]
    assert "third" in updated["compact_summary"]
    assert "first" not in updated["compact_summary"]
    assert "fact digest" in updated["compact_summary"]


def test_summarizer_truncates_long_summary() -> None:
    state = {
        "topic": "topic",
        "transcript": [_turn("Debater A", "x" * 500)],
        "search_results": [],
        "compact_summary": "old " * 600,
        "turn_count": 1,
        "max_turns": 8,
        "next_action": "speak_a",
        "last_decision": {"action": "speak_a", "reason": "x"},
        "final_summary": None,
        "markdown_path": "x.md",
        "session_id": "sid",
        "last_error": None,
    }

    summarizer = RuleBasedSummarizer(recent_turns=1, max_summary_chars=300)
    updated = summarizer_node(state, summarizer)

    assert len(updated["compact_summary"]) <= 300
