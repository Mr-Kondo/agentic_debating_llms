from __future__ import annotations

from app.graph import route_by_next_action


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
