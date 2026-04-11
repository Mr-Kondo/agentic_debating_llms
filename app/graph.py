"""LangGraph construction for the debate workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes.continuation_facilitator import continuation_facilitator_node
from app.nodes.debater import debater_a_node, debater_b_node
from app.nodes.facilitator import facilitator_node
from app.nodes.finalize import finalize_continuation_node, finalize_node, finish_node
from app.nodes.search import search_node
from app.nodes.summarizer import summarizer_node
from app.nodes.validator import validator_node
from app.state import DiscussionState


def route_by_next_action(state: DiscussionState) -> str:
    """Route to the next node based on state['next_action']."""
    action = state.get("next_action", "finish")
    if action == "speak_a":
        return "debater_a"
    if action == "speak_b":
        return "debater_b"
    if action == "search":
        return "search"
    return "finish"


def route_after_debater(state: DiscussionState) -> str:
    """Route debater output either to search or validator."""
    if state.get("next_action") == "search":
        return "search"
    return "validator"


def route_after_validator(state: DiscussionState) -> str:
    """Route validator output either to search or summarizer."""
    if state.get("next_action") == "search":
        return "search"
    return "summarizer"


def route_after_summarizer(state: DiscussionState) -> str:
    """Route to continuation facilitator when in continuation mode, else regular facilitator."""
    if state.get("continuation_mode", False):
        return "continuation_facilitator"
    return "facilitator"


def route_after_finalize(state: DiscussionState) -> str:
    """Route to continuation facilitator if continuation rounds are configured."""
    if state.get("continuation_max_turns", 0) > 0:
        return "continuation_facilitator"
    return END


def route_continuation_next_action(state: DiscussionState) -> str:
    """Route continuation facilitator action to debater or search or conclude."""
    action = state.get("next_action", "conclude")
    if action == "speak_a":
        return "debater_a"
    if action == "speak_b":
        return "debater_b"
    if action == "search":
        return "search"
    return "finalize_continuation"


def build_graph(services):
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(DiscussionState)

    graph.add_node("facilitator", lambda s: facilitator_node(s, services))
    graph.add_node("debater_a", lambda s: debater_a_node(s, services))
    graph.add_node("debater_b", lambda s: debater_b_node(s, services))
    graph.add_node("validator", lambda s: validator_node(s, services))
    graph.add_node("search", lambda s: search_node(s, services))
    graph.add_node("summarizer", lambda s: summarizer_node(s, services.summarizer))
    graph.add_node("finish", lambda s: finish_node(s, services))
    graph.add_node("finalize", lambda s: finalize_node(s, services))
    graph.add_node("continuation_facilitator", lambda s: continuation_facilitator_node(s, services))
    graph.add_node("finalize_continuation", lambda s: finalize_continuation_node(s, services))

    graph.add_edge(START, "facilitator")
    graph.add_conditional_edges(
        "facilitator",
        route_by_next_action,
        {
            "debater_a": "debater_a",
            "debater_b": "debater_b",
            "search": "search",
            "finish": "finish",
        },
    )
    graph.add_conditional_edges(
        "debater_a",
        route_after_debater,
        {
            "search": "search",
            "validator": "validator",
        },
    )
    graph.add_conditional_edges(
        "debater_b",
        route_after_debater,
        {
            "search": "search",
            "validator": "validator",
        },
    )
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "search": "search",
            "summarizer": "summarizer",
        },
    )
    graph.add_edge("search", "summarizer")
    graph.add_conditional_edges(
        "summarizer",
        route_after_summarizer,
        {
            "facilitator": "facilitator",
            "continuation_facilitator": "continuation_facilitator",
        },
    )
    graph.add_edge("finish", "finalize")
    graph.add_conditional_edges(
        "finalize",
        route_after_finalize,
        {
            "continuation_facilitator": "continuation_facilitator",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "continuation_facilitator",
        route_continuation_next_action,
        {
            "debater_a": "debater_a",
            "debater_b": "debater_b",
            "search": "search",
            "finalize_continuation": "finalize_continuation",
        },
    )
    graph.add_edge("finalize_continuation", END)

    return graph.compile()
