"""CLI entrypoint for running a local LLM debate session."""

from __future__ import annotations

import argparse
import sys

from app.config import load_config
from app.graph import build_graph
from app.services.session_service import initialize_session


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run local LLM debate session")
    parser.add_argument("--topic", required=True, help="Debate topic")
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

    try:
        services, initial_state = initialize_session(
            config=config,
            topic=args.topic,
            preload_models=not args.no_preload,
        )

        graph = build_graph(services)
        final_state = graph.invoke(initial_state)
    except RuntimeError as exc:
        print("Startup failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("Tip: try '--no-preload' for initial diagnostics.", file=sys.stderr)
        raise SystemExit(1) from exc

    print("=== Session Completed ===")
    print(f"Session ID: {final_state['session_id']}")
    print(f"Markdown log: {final_state['markdown_path']}")
    print(f"Final summary:\n{final_state['final_summary']}")


if __name__ == "__main__":
    main()
