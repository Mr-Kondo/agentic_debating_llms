"""Search node implementation."""

from __future__ import annotations

from pathlib import Path

from app.schemas import DiscussionTurn, SearchResult
from app.state import DiscussionState
from app.utils.retry import SearchRetryPolicy, run_with_search_retry
from app.utils.time_utils import now_utc


def search_node(state: DiscussionState, services) -> dict:
    """Run external search CLI and append result (including failures)."""
    decision = state["last_decision"]
    if isinstance(decision, dict):
        query = decision.get("search_query")
    else:
        query = decision.search_query

    query = query or state["topic"]

    try:
        with services.langfuse.span("search", input_data={"query": query}) as _:
            result = run_with_search_retry(
                operation=lambda: services.search_service.run(query=query),
                policy=SearchRetryPolicy(),
            )
            last_error = None
    except Exception as exc:
        services.langfuse.log_error(str(exc), error_type=type(exc).__name__)
        if hasattr(exc, "result"):
            result = exc.result
        else:
            result = SearchResult(
                query=query,
                stdout="",
                stderr=str(exc),
                returncode=1,
                digest=f"search failed: {exc}",
            )
        last_error = f"search_error:{exc}"

    search_results = [*state["search_results"], result]
    markdown_path = Path(state["markdown_path"])
    services.markdown_logger.append_search_result(path=markdown_path, result=result)

    turn_note = f"Search query='{query}' returncode={result.returncode} digest={result.digest}"
    turn = DiscussionTurn(role="Search", content=turn_note, timestamp=now_utc())
    return {
        "search_results": search_results,
        "turn_count": state["turn_count"] if state.get("continuation_mode", False) else state["turn_count"] + 1,
        "last_error": last_error,
        "transcript": [*state["transcript"], turn],
    }
