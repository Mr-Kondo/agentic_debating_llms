"""Continuation facilitator node — post-conclusion challenger."""

from __future__ import annotations

from pathlib import Path

from app.prompts import CONTINUATION_FACILITATOR_SYSTEM_PROMPT, build_continuation_facilitator_prompt
from app.schemas import ContinuationDecision
from app.state import DiscussionState, get_recent_turns
from app.utils.retry import LLMRetryPolicy, run_with_llm_retry


def continuation_facilitator_node(state: DiscussionState, services) -> dict:
    """Choose the next continuation action, targeting blind spots in the conclusion."""
    config = services.config
    cont_turn = state.get("continuation_turn_count", 0)
    cont_max = state.get("continuation_max_turns", 0)

    if cont_turn >= cont_max:
        decision = ContinuationDecision(
            action="conclude",
            reason="Reached continuation round limit.",
            conclude_reason="continuation_max_turns_reached",
        )
        return {
            "last_decision": decision.model_dump(),
            "next_action": "conclude",
        }

    prompt = build_continuation_facilitator_prompt(
        topic=state["topic"],
        final_summary=state.get("final_summary") or "",
        compact_summary=state["compact_summary"],
        recent_turns=get_recent_turns(state, config.recent_context_turns),
        continuation_turn_count=cont_turn,
        continuation_max_turns=cont_max,
    )

    def invoke_decision() -> ContinuationDecision:
        return services.ollama_client.generate_structured(
            model=config.facilitator_model,
            system_prompt=CONTINUATION_FACILITATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            schema_model=ContinuationDecision,
            keep_alive=config.model_keep_alive,
        )

    def recover_model() -> None:
        services.model_manager.ensure_loaded(config.facilitator_model, warmup=True)

    try:
        with services.langfuse.span(
            "continuation_facilitator", input_data={"prompt": prompt}
        ) as span:
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
                metadata={"node": "continuation_facilitator", "cont_turn": cont_turn},
            )
        next_cont_turn = cont_turn + 1
    except Exception as exc:
        services.langfuse.log_error(str(exc), error_type=type(exc).__name__)
        decision = ContinuationDecision(
            action="conclude",
            reason="Continuation facilitator failed; concluding safely.",
            conclude_reason="continuation_facilitator_failure",
        )
        last_error = f"continuation_facilitator_error:{exc}"
        next_cont_turn = cont_turn
    else:
        last_error = None

    markdown_path = Path(state["markdown_path"])
    services.markdown_logger.append_continuation_decision(
        path=markdown_path,
        decision=decision,
    )

    # Map continuation actions to existing debater node keys
    action_map = {
        "continue_a": "speak_a",
        "continue_b": "speak_b",
        "search": "search",
        "conclude": "conclude",
    }
    next_action = action_map.get(decision.action, "conclude")

    return {
        "last_decision": decision.model_dump(),
        "next_action": next_action,
        "continuation_turn_count": next_cont_turn,
        "last_error": last_error,
    }
