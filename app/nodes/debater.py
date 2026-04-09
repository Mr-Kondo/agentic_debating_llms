"""Debater node implementations."""

from __future__ import annotations

from pathlib import Path

from app.prompts import (
    DEBATER_A_SYSTEM_PROMPT,
    DEBATER_B_SYSTEM_PROMPT,
    build_debater_prompt,
)
from app.schemas import DebaterResponse, DiscussionTurn
from app.state import DiscussionState, get_recent_turns, latest_search_digest
from app.utils.retry import LLMRetryPolicy, run_with_llm_retry
from app.utils.time_utils import now_utc


def _debater_node(state: DiscussionState, services, speaker: str) -> dict:
    config = services.config
    decision = state["last_decision"]
    if isinstance(decision, dict):
        focus_instruction = decision.get("focus_instruction")
    else:
        focus_instruction = decision.focus_instruction
    focus_instruction = focus_instruction or "Add the most useful next argument."

    system_prompt = DEBATER_A_SYSTEM_PROMPT if speaker == "A" else DEBATER_B_SYSTEM_PROMPT
    model = config.debater_a_model if speaker == "A" else config.debater_b_model

    prompt = build_debater_prompt(
        speaker=speaker,
        topic=state["topic"],
        focus_instruction=focus_instruction,
        compact_summary=state["compact_summary"],
        recent_turns=get_recent_turns(state, config.recent_context_turns),
        search_digest=latest_search_digest(state),
    )

    def invoke_response() -> DebaterResponse:
        response = services.ollama_client.generate_structured(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            schema_model=DebaterResponse,
            keep_alive=config.model_keep_alive,
        )
        if response.speaker != speaker:
            return DebaterResponse(
                speaker=speaker,
                claim=response.claim,
                stance_summary=response.stance_summary,
                confidence=response.confidence,
            )
        return response

    def recover_model() -> None:
        services.model_manager.ensure_loaded(model, warmup=True)

    with services.langfuse.span(f"debater_{speaker.lower()}", input_data={"prompt": prompt}) as span:
        response = run_with_llm_retry(
            operation=invoke_response,
            policy=LLMRetryPolicy(),
            on_model_not_loaded=recover_model,
        )
        services.langfuse.log_generation(
            span=span,
            model=model,
            prompt=prompt,
            completion=response.model_dump_json(),
            metadata={"node": f"debater_{speaker.lower()}"},
        )

    turn = DiscussionTurn(
        role=f"Debater {speaker}",
        content=response.claim,
        timestamp=now_utc(),
    )
    transcript = [*state["transcript"], turn]

    markdown_path = Path(state["markdown_path"])
    services.markdown_logger.append_debater_utterance(path=markdown_path, response=response)

    return {
        "transcript": transcript,
        "turn_count": state["turn_count"] + 1,
        "last_error": None,
    }


def debater_a_node(state: DiscussionState, services) -> dict:
    """Debater A node."""
    return _debater_node(state=state, services=services, speaker="A")


def debater_b_node(state: DiscussionState, services) -> dict:
    """Debater B node."""
    return _debater_node(state=state, services=services, speaker="B")
