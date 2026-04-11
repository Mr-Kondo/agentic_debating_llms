"""Validator node implementation."""

from __future__ import annotations

from pathlib import Path

from app.prompts import VALIDATOR_SYSTEM_PROMPT, build_validator_prompt
from app.schemas import ValidatorFeedback
from app.state import DiscussionState, get_recent_turns
from app.utils.retry import LLMRetryPolicy, run_with_llm_retry


def validator_node(state: DiscussionState, services) -> dict:
    """Validate the latest debater utterance with validator model."""
    transcript = state.get("transcript", [])
    if not transcript:
        return {}

    latest = transcript[-1]
    role = latest.role if hasattr(latest, "role") else latest.get("role", "")
    content = latest.content if hasattr(latest, "content") else latest.get("content", "")
    if not role.startswith("Debater"):
        return {}

    prompt = build_validator_prompt(
        topic=state["topic"],
        speaker=role,
        claim=content,
        compact_summary=state["compact_summary"],
        recent_turns=get_recent_turns(state, services.config.recent_context_turns),
    )

    def invoke_feedback() -> ValidatorFeedback:
        return services.ollama_client.generate_structured(
            model=services.config.validator_model,
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            schema_model=ValidatorFeedback,
            keep_alive=services.config.model_keep_alive,
        )

    def recover_model() -> None:
        services.model_manager.ensure_loaded(services.config.validator_model, warmup=True)

    with services.langfuse.span("validator", input_data={"prompt": prompt}) as span:
        try:
            feedback = run_with_llm_retry(
                operation=invoke_feedback,
                policy=LLMRetryPolicy(),
                on_model_not_loaded=recover_model,
            )
        except Exception as exc:
            services.langfuse.log_error(str(exc), error_type=type(exc).__name__, span=span)
            feedback = ValidatorFeedback(
                is_valid=False,
                confidence=0.0,
                issues=f"validator_failure:{type(exc).__name__}",
                improvement="Continue debate with caution and verify evidence via search.",
            )
        else:
            services.langfuse.log_generation(
                span=span,
                model=services.config.validator_model,
                prompt=prompt,
                completion=feedback.model_dump_json(),
                metadata={"node": "validator"},
                usage_details=getattr(services.ollama_client, "_last_usage", None),
            )

    markdown_path = Path(state["markdown_path"])
    services.markdown_logger.append_validator_feedback(path=markdown_path, feedback=feedback)
    validation_log = [*state.get("validation_log", []), feedback]

    if feedback.needs_search and state.get("search_enabled", True):
        return {
            "validation_log": validation_log,
            "next_action": "search",
            "last_decision": {
                "action": "search",
                "reason": feedback.search_reason or "Validator requested evidence check.",
                "search_query": feedback.search_query,
                "request_source": "validator",
            },
            "last_error": None,
        }

    return {
        "validation_log": validation_log,
        "next_action": "facilitator",
        "last_error": None,
    }
