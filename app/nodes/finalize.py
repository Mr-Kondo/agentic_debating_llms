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
    highlights = []
    for item in state.get("validation_log", []):
        if isinstance(item, dict):
            highlights.append(str(item.get("issues", "")))
        else:
            highlights.append(item.issues)
    result_path = services.markdown_logger.write_result_snapshot(
        session_id=state["session_id"],
        topic=state["topic"],
        final_summary=summary,
        input_sources=state.get("input_sources", []),
        validation_highlights=[h for h in highlights if h][:3],
    )

    continuation_max_turns = state.get("continuation_max_turns", 0)
    return {
        "final_summary": summary,
        "result_markdown_path": str(result_path),
        "continuation_mode": continuation_max_turns > 0,
    }


def finalize_continuation_node(state: DiscussionState, services) -> dict:
    """Append continuation summary to markdown and update the result snapshot."""
    markdown_path = Path(state["markdown_path"])
    continuation_note = (
        f"## Continuation Phase Summary\n\n"
        f"Continuation rounds completed: {state.get('continuation_turn_count', 0)}\n\n"
        f"{state['compact_summary']}\n"
    )
    services.markdown_logger._append(markdown_path, continuation_note)

    highlights = []
    for item in state.get("validation_log", []):
        if isinstance(item, dict):
            highlights.append(str(item.get("issues", "")))
        else:
            highlights.append(item.issues)
    result_path = services.markdown_logger.write_result_snapshot(
        session_id=state["session_id"],
        topic=state["topic"],
        final_summary=state.get("final_summary") or state["compact_summary"],
        input_sources=state.get("input_sources", []),
        validation_highlights=[h for h in highlights if h][:3],
    )
    return {"result_markdown_path": str(result_path), "continuation_mode": False}
