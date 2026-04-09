from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas import DebaterResponse, DiscussionTurn, FacilitatorDecision, SearchResult, ValidatorFeedback


def test_facilitator_decision_valid() -> None:
    decision = FacilitatorDecision(
        action="search",
        reason="Need external evidence",
        search_query="latest benchmark local llm",
    )
    assert decision.action == "search"


def test_facilitator_decision_invalid_action() -> None:
    with pytest.raises(ValidationError):
        FacilitatorDecision(action="invalid", reason="x")


def test_debater_response_confidence_range() -> None:
    with pytest.raises(ValidationError):
        DebaterResponse(
            speaker="A",
            claim="claim",
            stance_summary="summary",
            confidence=1.5,
        )


def test_search_result_roundtrip() -> None:
    result = SearchResult(
        query="topic",
        stdout="a",
        stderr="",
        returncode=0,
        digest="digest",
    )
    dumped = result.model_dump()
    reloaded = SearchResult.model_validate(dumped)
    assert reloaded.query == "topic"


def test_discussion_turn_timestamp() -> None:
    turn = DiscussionTurn(role="Debater A", content="point", timestamp=datetime.now(timezone.utc))
    assert turn.timestamp.tzinfo is not None


def test_validator_feedback_valid() -> None:
    feedback = ValidatorFeedback(
        is_valid=True,
        confidence=0.8,
        issues="minor caveat",
        improvement="add one concrete example",
    )
    assert feedback.is_valid is True


def test_validator_feedback_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        ValidatorFeedback(
            is_valid=False,
            confidence=1.2,
            issues="issue",
            improvement="fix",
        )
