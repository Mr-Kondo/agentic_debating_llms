from __future__ import annotations

from app.graph import route_by_next_action, route_after_summarizer, route_after_finalize, route_continuation_next_action


def test_route_speak_a() -> None:
    state = {"next_action": "speak_a"}
    assert route_by_next_action(state) == "debater_a"


def test_route_speak_b() -> None:
    state = {"next_action": "speak_b"}
    assert route_by_next_action(state) == "debater_b"


def test_route_search() -> None:
    state = {"next_action": "search"}
    assert route_by_next_action(state) == "search"


def test_route_default_finalize() -> None:
    state = {"next_action": "finish"}
    assert route_by_next_action(state) == "finish"


def test_route_unknown_finalize() -> None:
    state = {"next_action": "unknown"}
    assert route_by_next_action(state) == "finish"


def test_route_finish_explicit() -> None:
    state = {"next_action": "finish"}
    assert route_by_next_action(state) == "finish"


# Continuation routing tests

def test_route_after_summarizer_normal_mode() -> None:
    state = {"continuation_mode": False}
    assert route_after_summarizer(state) == "facilitator"


def test_route_after_summarizer_continuation_mode() -> None:
    state = {"continuation_mode": True}
    assert route_after_summarizer(state) == "continuation_facilitator"


def test_route_after_summarizer_default_no_continuation() -> None:
    state = {}
    assert route_after_summarizer(state) == "facilitator"


def test_route_after_finalize_no_continuation() -> None:
    from langgraph.graph import END
    state = {"continuation_max_turns": 0}
    assert route_after_finalize(state) == END


def test_route_after_finalize_with_continuation() -> None:
    state = {"continuation_max_turns": 3}
    assert route_after_finalize(state) == "continuation_facilitator"


def test_route_after_finalize_default_no_continuation() -> None:
    from langgraph.graph import END
    state = {}
    assert route_after_finalize(state) == END


def test_route_continuation_speak_a() -> None:
    state = {"next_action": "speak_a"}
    assert route_continuation_next_action(state) == "debater_a"


def test_route_continuation_speak_b() -> None:
    state = {"next_action": "speak_b"}
    assert route_continuation_next_action(state) == "debater_b"


def test_route_continuation_search() -> None:
    state = {"next_action": "search"}
    assert route_continuation_next_action(state) == "search"


def test_route_continuation_conclude() -> None:
    state = {"next_action": "conclude"}
    assert route_continuation_next_action(state) == "finalize_continuation"


def test_route_continuation_default_conclude() -> None:
    state = {"next_action": "unknown"}
    assert route_continuation_next_action(state) == "finalize_continuation"
