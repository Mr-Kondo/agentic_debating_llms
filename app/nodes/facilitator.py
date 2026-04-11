"""Facilitator node implementation."""

from __future__ import annotations

from pathlib import Path

from app.prompts import FACILITATOR_SYSTEM_PROMPT, build_facilitator_prompt
from app.schemas import FacilitatorDecision
from app.state import DiscussionState, count_debater_turns, get_recent_turns, latest_search_digest
from app.utils.retry import LLMRetryPolicy, run_with_llm_retry


def facilitator_node(state: DiscussionState, services) -> dict:
    """Choose the next action using structured output with safety fallback."""
    config = services.config

    if state["turn_count"] >= state["max_turns"]:
        decision = FacilitatorDecision(
            action="finish",
            reason="Reached max_turns limit.",
            terminate_reason="max_turns_reached",
        )
        services.markdown_logger.append_facilitator_decision(
            path=Path(state["markdown_path"]),
            decision=decision,
        )
        return {
            "last_decision": decision,
            "next_action": decision.action,
        }

    a_count, b_count = count_debater_turns(state)
    prompt = build_facilitator_prompt(
        topic=state["topic"],
        compact_summary=state["compact_summary"],
        recent_turns=get_recent_turns(state, config.recent_context_turns),
        search_digest=latest_search_digest(state),
        turn_count=state["turn_count"],
        max_turns=state["max_turns"],
        search_available=state.get("search_enabled", True),
        a_count=a_count,
        b_count=b_count,
    )

    def invoke_decision() -> FacilitatorDecision:
        return services.ollama_client.generate_structured(
            model=config.facilitator_model,
            system_prompt=FACILITATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            schema_model=FacilitatorDecision,
            keep_alive=config.model_keep_alive,
        )

    def recover_model() -> None:
        services.model_manager.ensure_loaded(config.facilitator_model, warmup=True)

    try:
        with services.langfuse.span("facilitator", input_data={"prompt": prompt}) as span:
            decision = run_with_llm_retry(
                operation=invoke_decision,
                policy=LLMRetryPolicy(),
                on_model_not_loaded=recover_model,
            )
            services.langfuse.log_generation(
                span=span,
                model=config.facilitator_model,
                prompt=prompt,
                completion=decision.model_dump_json(),
                metadata={"node": "facilitator"},
                usage_details=getattr(services.ollama_client, "_last_usage", None),
            )
    except Exception as exc:
        services.langfuse.log_error(str(exc), error_type=type(exc).__name__)
        decision = FacilitatorDecision(
            action="finish",
            reason="Facilitator failed repeatedly; using safety fallback.",
            terminate_reason="facilitator_failure",
        )
        last_error = f"facilitator_error:{exc}"
    else:
        last_error = None

    if not state.get("search_enabled", True) and decision.action == "search":
        fallback_action = "speak_a" if state["turn_count"] % 2 == 0 else "speak_b"
        decision = FacilitatorDecision(
            action=fallback_action,
            reason=(f"Search is unavailable; replaced search action with {fallback_action} to keep debate progressing."),
            focus_instruction=decision.focus_instruction,
        )

    # Speaker balance guard: ensure Debater B gets a turn when A has spoken but B hasn't.
    a_count, b_count = count_debater_turns(state)
    if b_count == 0 and a_count > 0 and decision.action in ("speak_a", "search", "finish"):
        decision = FacilitatorDecision(
            action="speak_b",
            reason="Speaker balance correction: Debater B has not spoken yet.",
            focus_instruction=decision.focus_instruction or "Present your opening argument.",
        )

    markdown_path = Path(state["markdown_path"])
    services.markdown_logger.append_facilitator_decision(path=markdown_path, decision=decision)

    return {
        "last_decision": decision,
        "next_action": decision.action,
        "last_error": last_error,
    }
