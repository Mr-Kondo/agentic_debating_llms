"""Prompt templates used by facilitator and debaters."""

from __future__ import annotations

from app.schemas import DiscussionTurn


FACILITATOR_SYSTEM_PROMPT = """You are the facilitator of a structured debate.
Decide the next action among: speak_a, speak_b, search, finish.
Output must follow the provided JSON schema exactly.
Favor concise progress, avoid loops, and finish when enough evidence exists.
"""

DEBATER_A_SYSTEM_PROMPT = """You are Debater A.
Model persona: analytical and practical.
Produce concise, non-redundant arguments.
Output must follow JSON schema exactly.
"""

DEBATER_B_SYSTEM_PROMPT = """You are Debater B.
Model persona: critical and evidence-driven.
Produce concise, non-redundant arguments.
Output must follow JSON schema exactly.
"""

FINALIZER_SYSTEM_PROMPT = """You are a neutral summarizer.
Write a short final summary with key claims from both sides and practical takeaways.
"""


def _render_recent_turns(recent_turns: list[DiscussionTurn]) -> str:
    if not recent_turns:
        return "(no turns yet)"
    lines: list[str] = []
    for turn in recent_turns:
        lines.append(f"- {turn.timestamp.isoformat()} [{turn.role}] {turn.content}")
    return "\n".join(lines)


def build_facilitator_prompt(
    topic: str,
    compact_summary: str,
    recent_turns: list[DiscussionTurn],
    search_digest: str,
    turn_count: int,
    max_turns: int,
) -> str:
    """Create facilitator prompt from compressed context only."""
    recent = _render_recent_turns(recent_turns)
    return (
        f"Topic: {topic}\n"
        f"Turn: {turn_count}/{max_turns}\n"
        f"Compact Summary:\n{compact_summary or '(empty)'}\n\n"
        f"Latest Search Digest:\n{search_digest or '(none)'}\n\n"
        f"Recent Turns:\n{recent}\n\n"
        "Decide next action.\n"
        "- Use search when concrete facts are missing.\n"
        "- Use finish when decision quality is sufficient or max_turns is near.\n"
        "- Provide focus_instruction for debater actions and search_query for search action."
    )


def build_debater_prompt(
    speaker: str,
    topic: str,
    focus_instruction: str,
    compact_summary: str,
    recent_turns: list[DiscussionTurn],
    search_digest: str,
) -> str:
    """Create debater prompt with bounded context."""
    recent = _render_recent_turns(recent_turns)
    return (
        f"You are speaker {speaker}.\n"
        f"Topic: {topic}\n"
        f"Focus instruction: {focus_instruction or 'Add the most valuable next point.'}\n"
        f"Compact summary:\n{compact_summary or '(empty)'}\n\n"
        f"Latest search digest:\n{search_digest or '(none)'}\n\n"
        f"Recent turns:\n{recent}\n\n"
        "Respond with one high-value claim and avoid repeating earlier points."
    )


def build_finalizer_prompt(topic: str, compact_summary: str, recent_turns: list[DiscussionTurn]) -> str:
    """Create final summarization prompt."""
    recent = _render_recent_turns(recent_turns)
    return (
        f"Topic: {topic}\n"
        f"Compact summary:\n{compact_summary or '(empty)'}\n\n"
        f"Recent turns:\n{recent}\n\n"
        "Produce a practical final summary in Japanese with:\n"
        "1) key arguments from A and B\n"
        "2) unresolved points\n"
        "3) recommendation or next actions"
    )
