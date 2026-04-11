"""Search node implementation."""

from __future__ import annotations

from pathlib import Path

from app.schemas import DiscussionTurn, SearchResult
from app.services.search_query_optimizer import optimize_search_query
from app.services.search_service import SearchCLIError, SearchTimeoutError
from app.state import DiscussionState
from app.utils.retry import SearchRetryPolicy, run_with_search_retry
from app.utils.time_utils import now_utc


def _classify_search_error(exc: Exception, result: SearchResult | None = None) -> str:
    if isinstance(exc, SearchTimeoutError):
        return "search_timeout"
    if isinstance(exc, SearchCLIError) and result is not None:
        stderr_lower = result.stderr.lower()
        if result.returncode == 127:
            return "search_cli_not_found"
        if result.returncode in (2, 64):
            return "search_invalid_template_or_args"
        if "missing option" in stderr_lower or "no such command" in stderr_lower or "invalid value" in stderr_lower:
            return "search_invalid_template_or_args"
        return "search_runtime_failure"
    return "search_error"


def search_node(state: DiscussionState, services) -> dict:
    """Run external search CLI and append result (including failures)."""
    decision = state["last_decision"]
    if isinstance(decision, dict):
        query = decision.get("search_query")
        request_source = decision.get("request_source", "facilitator")
    else:
        query = decision.search_query
        request_source = "facilitator"

    query = query or state["topic"]
    optimizer_mode = getattr(getattr(services, "config", None), "search_query_optimizer", "none")
    effective_query = optimize_search_query(query=query, topic=state["topic"], mode=optimizer_mode)

    if not state.get("search_enabled", True):
        status_message = state.get("search_status_message") or "Search is disabled in this session."
        result = SearchResult(
            query=effective_query,
            stdout="",
            stderr=status_message,
            returncode=127,
            digest=f"search unavailable: {status_message}",
        )
        services.langfuse.log_error(status_message, error_type="SearchUnavailable")
        search_results = [*state["search_results"], result]
        markdown_path = Path(state["markdown_path"])
        services.markdown_logger.append_search_result(path=markdown_path, result=result)
        turn_note = (
            f"Search skipped source='{request_source}' query='{effective_query}' "
            f"returncode={result.returncode} reason={status_message}"
        )
        turn = DiscussionTurn(role="Search", content=turn_note, timestamp=now_utc())
        return {
            "search_results": search_results,
            "turn_count": state["turn_count"] if state.get("continuation_mode", False) else state["turn_count"] + 1,
            "last_error": "search_unavailable",
            "transcript": [*state["transcript"], turn],
        }

    try:
        with services.langfuse.span("search", input_data={"query": query}) as _:
            result = run_with_search_retry(
                operation=lambda: services.search_service.run(query=effective_query),
                policy=SearchRetryPolicy(),
            )
            last_error = None
    except Exception as exc:
        services.langfuse.log_error(str(exc), error_type=type(exc).__name__)
        if hasattr(exc, "result"):
            result = exc.result
        else:
            result = SearchResult(
                query=effective_query,
                stdout="",
                stderr=str(exc),
                returncode=1,
                digest=f"search failed: {exc}",
            )
        last_error = _classify_search_error(exc, result)

    search_results = [*state["search_results"], result]
    markdown_path = Path(state["markdown_path"])
    services.markdown_logger.append_search_result(path=markdown_path, result=result)

    turn_note = f"Search source='{request_source}' query='{query}' returncode={result.returncode} digest={result.digest}"
    if effective_query != query:
        turn_note = (
            f"Search source='{request_source}' original_query='{query}' optimized_query='{effective_query}' "
            f"returncode={result.returncode} digest={result.digest}"
        )
    else:
        turn_note = (
            f"Search source='{request_source}' query='{effective_query}' returncode={result.returncode} digest={result.digest}"
        )
    turn = DiscussionTurn(role="Search", content=turn_note, timestamp=now_utc())
    return {
        "search_results": search_results,
        "turn_count": state["turn_count"] if state.get("continuation_mode", False) else state["turn_count"] + 1,
        "last_error": last_error,
        "transcript": [*state["transcript"], turn],
    }
