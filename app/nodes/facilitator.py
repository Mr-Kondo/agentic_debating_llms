"""Facilitator node implementation."""

from __future__ import annotations

from pathlib import Path

from app.prompts import FACILITATOR_SYSTEM_PROMPT, build_facilitator_prompt
from app.schemas import FacilitatorDecision
from app.state import DiscussionState, get_recent_turns, latest_search_digest
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

    prompt = build_facilitator_prompt(
        topic=state["topic"],
        compact_summary=state["compact_summary"],
        recent_turns=get_recent_turns(state, config.recent_context_turns),
        search_digest=latest_search_digest(state),
        turn_count=state["turn_count"],
        max_turns=state["max_turns"],
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

    markdown_path = Path(state["markdown_path"])
    services.markdown_logger.append_facilitator_decision(path=markdown_path, decision=decision)

    return {
        "last_decision": decision,
        "next_action": decision.action,
        "last_error": last_error,
    }
