"""Tests for facilitator node speaker-balance guard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.nodes.facilitator import facilitator_node
from app.schemas import DiscussionTurn, FacilitatorDecision
from app.utils.time_utils import now_utc


def _make_turn(role: str, content: str = "some argument") -> DiscussionTurn:
    return DiscussionTurn(role=role, content=content, timestamp=now_utc())


def _base_state(**overrides) -> dict:
    state = {
        "topic": "test topic",
        "transcript": [],
        "compact_summary": "",
        "turn_count": 1,
        "max_turns": 10,
        "search_results": [],
        "search_enabled": True,
        "search_status_message": None,
        "validation_log": [],
        "markdown_path": "/tmp/debate.md",
        "session_id": "test-session",
        "last_error": None,
        "last_decision": None,
        "next_action": None,
        "final_summary": None,
        "result_markdown_path": None,
        "input_sources": [],
        "continuation_mode": False,
        "continuation_turn_count": 0,
        "continuation_max_turns": 0,
    }
    state.update(overrides)
    return state


def _make_services(decision: FacilitatorDecision) -> MagicMock:
    """Build a minimal services mock that returns the given FacilitatorDecision."""
    services = MagicMock()
    services.config.facilitator_model = "test-model"
    services.config.recent_context_turns = 4
    services.config.model_keep_alive = None
    services.ollama_client.generate_structured.return_value = decision
    services.ollama_client._last_usage = None
    services.langfuse.span.return_value.__enter__ = lambda s, *_: MagicMock()
    services.langfuse.span.return_value.__exit__ = MagicMock(return_value=False)
    services.markdown_logger.append_facilitator_decision = MagicMock()
    return services


class TestSpeakerBalanceGuard:
    def test_speak_b_forced_when_b_has_never_spoken(self):
        """If facilitator says speak_a but B has never spoken and A has, override to speak_b."""
        state = _base_state(
            transcript=[_make_turn("Debater A", "A's first claim")],
        )
        llm_decision = FacilitatorDecision(action="speak_a", reason="I want A again")
        services = _make_services(llm_decision)

        with patch("app.nodes.facilitator.run_with_llm_retry", side_effect=lambda operation, **kw: operation()):
            result = facilitator_node(state, services)  # type: ignore[arg-type]

        assert result["next_action"] == "speak_b"
        assert "balance" in result["last_decision"].reason.lower()

    def test_finish_overridden_when_b_has_never_spoken(self):
        """finish action is also overridden when B hasn't spoken yet."""
        state = _base_state(
            transcript=[_make_turn("Debater A")],
        )
        llm_decision = FacilitatorDecision(action="finish", reason="done")
        services = _make_services(llm_decision)

        with patch("app.nodes.facilitator.run_with_llm_retry", side_effect=lambda operation, **kw: operation()):
            result = facilitator_node(state, services)  # type: ignore[arg-type]

        assert result["next_action"] == "speak_b"

    def test_speak_b_not_overridden_when_b_already_spoke(self):
        """No override when B has at least one recorded turn."""
        state = _base_state(
            transcript=[
                _make_turn("Debater A"),
                _make_turn("Debater B"),
            ],
        )
        llm_decision = FacilitatorDecision(action="speak_a", reason="A's turn")
        services = _make_services(llm_decision)

        with patch("app.nodes.facilitator.run_with_llm_retry", side_effect=lambda operation, **kw: operation()):
            result = facilitator_node(state, services)  # type: ignore[arg-type]

        assert result["next_action"] == "speak_a"

    def test_balance_guard_not_triggered_when_transcript_empty(self):
        """No B-balance override when nobody has spoken yet."""
        state = _base_state(transcript=[])
        llm_decision = FacilitatorDecision(action="speak_a", reason="opening")
        services = _make_services(llm_decision)

        with patch("app.nodes.facilitator.run_with_llm_retry", side_effect=lambda operation, **kw: operation()):
            result = facilitator_node(state, services)  # type: ignore[arg-type]

        assert result["next_action"] == "speak_a"

    def test_search_overridden_when_b_has_never_spoken(self):
        """search action is also corrected to speak_b when B has never spoken."""
        state = _base_state(
            transcript=[_make_turn("Debater A")],
            search_enabled=True,
        )
        llm_decision = FacilitatorDecision(action="search", reason="need facts", search_query="query")
        services = _make_services(llm_decision)

        with patch("app.nodes.facilitator.run_with_llm_retry", side_effect=lambda operation, **kw: operation()):
            result = facilitator_node(state, services)  # type: ignore[arg-type]

        assert result["next_action"] == "speak_b"
