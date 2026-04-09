"""CLI entrypoint for running a local LLM debate session."""

from __future__ import annotations

import argparse
import sys

from app.config import load_config
from app.graph import build_graph
from app.services.input_service import InputSourceError, load_input_payload
from app.services.session_service import initialize_session


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run local LLM debate session")
    parser.add_argument("--topic", required=False, help="Debate topic")
    parser.add_argument("--max-turns", type=int, default=None, help="Override max turns")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip preload and warmup of Ollama models",
    )
    return parser.parse_args()


def main() -> None:
    """Run a discussion session via LangGraph orchestration."""
    args = parse_args()
    config = load_config()

    if args.max_turns is not None:
        config.max_turns = args.max_turns

    resolved_topic = args.topic
    initial_compact_summary = ""
    input_sources: list[str] = []

    if not resolved_topic:
        try:
            payload = load_input_payload(config.input_dir_path)
        except InputSourceError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from exc
        resolved_topic = payload.topic
        initial_compact_summary = payload.details
        input_sources = payload.sources

    services = None
    final_state = None
    try:
        services, initial_state = initialize_session(
            config=config,
            topic=resolved_topic,
            initial_compact_summary=initial_compact_summary,
            input_sources=input_sources,
            preload_models=not args.no_preload,
        )

        graph = build_graph(services)
        final_state = graph.invoke(initial_state)
    except RuntimeError as exc:
        print("Startup failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("Tip: try '--no-preload' for initial diagnostics.", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        if services is not None:
            output = None
            if final_state is not None:
                output = {"final_summary": final_state.get("final_summary")}
            services.langfuse.end_trace(output=output)

    print("=== Session Completed ===")
    print(f"Session ID: {final_state['session_id']}")
    print(f"Markdown log: {final_state['markdown_path']}")
    if final_state.get("result_markdown_path"):
        print(f"Result markdown: {final_state['result_markdown_path']}")
    print(f"Final summary:\n{final_state['final_summary']}")


if __name__ == "__main__":
    main()
