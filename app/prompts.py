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
Output must follow JSON schema exactly. Reply with a single JSON object and nothing else.
If external facts are necessary, set needs_search=true with a concise search_query.
Example output:
{"speaker": "A", "claim": "...", "stance_summary": "...", "confidence": 0.8}
"""

DEBATER_B_SYSTEM_PROMPT = """You are Debater B.
Model persona: critical and evidence-driven.
Produce concise, non-redundant arguments.
Output must follow JSON schema exactly. Reply with a single JSON object and nothing else.
If external facts are necessary, set needs_search=true with a concise search_query.
Example output:
{"speaker": "B", "claim": "...", "stance_summary": "...", "confidence": 0.8}
"""

VALIDATOR_SYSTEM_PROMPT = """You are a debate quality validator.
Evaluate the latest debater claim for logical coherence, topical relevance, and practical usefulness.
Be strict but constructive.
Output must follow the JSON schema exactly.
If evidence is insufficient, set needs_search=true and include a concrete search_query.
"""

FINALIZER_SYSTEM_PROMPT = """You are a neutral summarizer.
Write a short final summary with key claims from both sides and practical takeaways.
"""

CONTINUATION_FACILITATOR_SYSTEM_PROMPT = """You are a post-conclusion challenger.
The debate has produced an initial conclusion. Your role is to direct further scrutiny by finding:
1. Logical gaps or unstated assumptions in the conclusion
2. Edge cases or exceptions not adequately covered
3. Practical implementation risks or failure modes
4. Counterexamples or dissenting perspectives worth exploring
Do NOT repeat points already made in the transcript. Focus on genuine blind spots.
Decide next action: continue_a, continue_b, search, or conclude.
Output must follow the provided JSON schema exactly.
"""


def _render_recent_turns(recent_turns: list[DiscussionTurn]) -> str:
    if not recent_turns:
        return "(no turns yet)"
    lines: list[str] = []
    for turn in recent_turns:
        if isinstance(turn, dict):
            ts = str(turn.get("timestamp", ""))
            role = str(turn.get("role", ""))
            content = str(turn.get("content", ""))
        else:
            ts = turn.timestamp.isoformat()
            role = turn.role
            content = turn.content
        lines.append(f"- {ts} [{role}] {content}")
    return "\n".join(lines)


def build_facilitator_prompt(
    topic: str,
    compact_summary: str,
    recent_turns: list[DiscussionTurn],
    search_digest: str,
    turn_count: int,
    max_turns: int,
    search_available: bool = True,
    a_count: int = 0,
    b_count: int = 0,
) -> str:
    """Create facilitator prompt from compressed context only."""
    recent = _render_recent_turns(recent_turns)
    search_rule = (
        "- Search is currently unavailable; do not choose search action.\n"
        if not search_available
        else "- Use search when concrete facts are missing.\n"
    )
    balance_note = f"Speaker balance: Debater A has {a_count} turn(s), Debater B has {b_count} turn(s).\n" + (
        "- Debater B has not spoken yet; prefer speak_b to ensure both sides are heard.\n"
        if b_count == 0 and a_count > 0
        else ""
    )

    return (
        f"Topic: {topic}\n"
        f"Turn: {turn_count}/{max_turns}\n"
        f"Compact Summary:\n{compact_summary or '(empty)'}\n\n"
        f"Latest Search Digest:\n{search_digest or '(none)'}\n\n"
        f"Recent Turns:\n{recent}\n\n"
        "Decide next action.\n"
        f"{balance_note}"
        f"{search_rule}"
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


def build_finalizer_prompt(
    topic: str,
    compact_summary: str,
    recent_turns: list[DiscussionTurn],
    a_count: int = 0,
    b_count: int = 0,
) -> str:
    """Create final summarization prompt."""
    recent = _render_recent_turns(recent_turns)
    absence_notes: list[str] = []
    if a_count == 0:
        absence_notes.append("NOTE: Debater A made no recorded arguments in this session.")
    if b_count == 0:
        absence_notes.append("NOTE: Debater B made no recorded arguments in this session.")
    absence_section = ("\n".join(absence_notes) + "\n") if absence_notes else ""

    return (
        f"Topic: {topic}\n"
        f"Speaker turns recorded — Debater A: {a_count}, Debater B: {b_count}\n"
        f"{absence_section}"
        f"Compact summary:\n{compact_summary or '(empty)'}\n\n"
        f"Recent turns:\n{recent}\n\n"
        "Produce a practical final summary in Japanese with:\n"
        "1) key arguments from A and B (if a speaker has no recorded turns, state that explicitly "
        "instead of generating placeholder text)\n"
        "2) unresolved points\n"
        "3) recommendation or next actions"
    )


def build_validator_prompt(
    *,
    topic: str,
    speaker: str,
    claim: str,
    compact_summary: str,
    recent_turns: list[DiscussionTurn],
) -> str:
    """Create validator prompt for the latest debater claim."""
    recent = _render_recent_turns(recent_turns)
    return (
        f"Topic: {topic}\n"
        f"Speaker: {speaker}\n"
        f"Latest claim:\n{claim}\n\n"
        f"Compact summary:\n{compact_summary or '(empty)'}\n\n"
        f"Recent turns:\n{recent}\n\n"
        "Judge whether the latest claim is valid and useful in this debate context. "
        "Provide concrete issues and one clear improvement suggestion."
    )


def build_continuation_facilitator_prompt(
    topic: str,
    final_summary: str,
    compact_summary: str,
    recent_turns: list[DiscussionTurn],
    continuation_turn_count: int,
    continuation_max_turns: int,
) -> str:
    """Create continuation facilitator prompt to challenge the existing conclusion."""
    recent = _render_recent_turns(recent_turns)
    return (
        f"Topic: {topic}\n"
        f"Continuation round: {continuation_turn_count + 1}/{continuation_max_turns}\n\n"
        f"Initial conclusion:\n{final_summary or '(none)'}\n\n"
        f"Discussion summary so far:\n{compact_summary or '(empty)'}\n\n"
        f"Recent turns:\n{recent}\n\n"
        "Identify a genuine blind spot, edge case, or counterexample NOT yet covered.\n"
        "- Use continue_a or continue_b to direct a debater to explore a specific gap.\n"
        "- Use search when concrete external evidence is needed.\n"
        "- Use conclude when no significant blind spots remain or rounds are exhausted.\n"
        "- Provide focus_instruction describing exactly what gap to explore.\n"
        "- Do NOT revisit points already discussed."
    )
