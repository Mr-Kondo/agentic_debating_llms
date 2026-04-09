"""Finalize node implementation."""

from __future__ import annotations

from pathlib import Path

from app.prompts import FINALIZER_SYSTEM_PROMPT, build_finalizer_prompt
from app.state import DiscussionState, get_recent_turns


def finish_node(state: DiscussionState, services) -> dict:
    """Lightweight transition node before finalization."""
    _ = services
    return {"next_action": "finish"}


def finalize_node(state: DiscussionState, services) -> dict:
    """Finalize discussion and write final markdown section."""
    prompt = build_finalizer_prompt(
        topic=state["topic"],
        compact_summary=state["compact_summary"],
        recent_turns=get_recent_turns(state, services.config.recent_context_turns),
    )

    try:
        with services.langfuse.span("finalize", input_data={"prompt": prompt}) as span:
            summary = services.ollama_client.generate_text(
                model=services.config.facilitator_model,
                system_prompt=FINALIZER_SYSTEM_PROMPT,
                user_prompt=prompt,
                keep_alive=services.config.model_keep_alive,
            )
            services.langfuse.log_generation(
                span=span,
                model=services.config.facilitator_model,
                prompt=prompt,
                completion=summary,
                metadata={"node": "finalize"},
            )
    except Exception as exc:
        services.langfuse.log_error(str(exc), error_type=type(exc).__name__)
        summary = state["compact_summary"] or "Final summary generation failed."

    markdown_path = Path(state["markdown_path"])
    services.markdown_logger.append_final_summary(path=markdown_path, summary=summary)
    services.langfuse.end_trace(output={"final_summary": summary})
    return {"final_summary": summary}
