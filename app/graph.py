"""LangGraph construction for the debate workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes.debater import debater_a_node, debater_b_node
from app.nodes.facilitator import facilitator_node
from app.nodes.finalize import finalize_node, finish_node
from app.nodes.search import search_node
from app.nodes.summarizer import summarizer_node
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


def build_graph(services):
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(DiscussionState)

    graph.add_node("facilitator", lambda s: facilitator_node(s, services))
    graph.add_node("debater_a", lambda s: debater_a_node(s, services))
    graph.add_node("debater_b", lambda s: debater_b_node(s, services))
    graph.add_node("search", lambda s: search_node(s, services))
    graph.add_node("summarizer", lambda s: summarizer_node(s, services.summarizer))
    graph.add_node("finish", lambda s: finish_node(s, services))
    graph.add_node("finalize", lambda s: finalize_node(s, services))

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
    graph.add_edge("debater_a", "summarizer")
    graph.add_edge("debater_b", "summarizer")
    graph.add_edge("search", "summarizer")
    graph.add_edge("summarizer", "facilitator")
    graph.add_edge("finish", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
