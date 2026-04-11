from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app.nodes.search import search_node
from app.schemas import SearchResult
from app.services.search_service import SearchCLIError


class _DummyDecision:
    search_query = "test query"


class _DummyLogger:
    def __init__(self) -> None:
        self.search_calls = 0

    def append_search_result(self, path: Path, result) -> None:
        _ = (path, result)
        self.search_calls += 1


class _DummyLangfuse:
    @contextmanager
    def span(self, name: str, input_data=None):
        _ = (name, input_data)
        yield None

    def log_error(self, message: str, error_type: str = "RuntimeError", span=None) -> None:
        _ = (message, error_type, span)


class _DummySearchService:
    def run(self, query: str):
        raise AssertionError("run() should not be called when search is disabled")


class _FailingSearchService:
    def run(self, query: str):
        _ = query
        result = SearchResult(
            query="q",
            stdout="",
            stderr="Error: Missing option '-q'",
            returncode=2,
            digest="returncode=2",
        )
        raise SearchCLIError("Search CLI exited with non-zero status (invalid_template_or_args).", result)


class _DummyServices:
    def __init__(self) -> None:
        self.markdown_logger = _DummyLogger()
        self.langfuse = _DummyLangfuse()
        self.search_service = _DummySearchService()
        self.config = type("_Cfg", (), {"search_query_optimizer": "none"})()


class _FailingServices:
    def __init__(self) -> None:
        self.markdown_logger = _DummyLogger()
        self.langfuse = _DummyLangfuse()
        self.search_service = _FailingSearchService()
        self.config = type("_Cfg", (), {"search_query_optimizer": "none"})()


class _SuccessfulSearchService:
    def __init__(self) -> None:
        self.last_query: str | None = None

    def run(self, query: str):
        self.last_query = query
        return SearchResult(query=query, stdout='[{"title":"x"}]', stderr="", returncode=0, digest="ok")


class _OptimizingServices:
    def __init__(self) -> None:
        self.markdown_logger = _DummyLogger()
        self.langfuse = _DummyLangfuse()
        self.search_service = _SuccessfulSearchService()
        self.config = type("_Cfg", (), {"search_query_optimizer": "dspy"})()


def test_search_node_skips_cli_when_search_disabled() -> None:
    services = _DummyServices()
    state = {
        "last_decision": _DummyDecision(),
        "topic": "topic",
        "search_enabled": False,
        "search_status_message": "Search is disabled: CLI missing",
        "search_results": [],
        "markdown_path": "logs/debate_test.md",
        "turn_count": 0,
        "continuation_mode": False,
        "transcript": [
            {
                "role": "Facilitator",
                "content": "search next",
                "timestamp": datetime.now(timezone.utc),
            }
        ],
    }

    updated = search_node(state, services)

    assert updated["last_error"] == "search_unavailable"
    assert len(updated["search_results"]) == 1
    assert updated["search_results"][0].returncode == 127
    assert services.markdown_logger.search_calls == 1


def test_search_node_sets_classified_last_error_on_invalid_args() -> None:
    services = _FailingServices()
    state = {
        "last_decision": _DummyDecision(),
        "topic": "topic",
        "search_enabled": True,
        "search_status_message": None,
        "search_results": [],
        "markdown_path": "logs/debate_test.md",
        "turn_count": 0,
        "continuation_mode": False,
        "transcript": [],
    }

    updated = search_node(state, services)

    assert updated["last_error"] == "search_invalid_template_or_args"
    assert updated["search_results"][0].returncode == 2
    assert services.markdown_logger.search_calls == 1


def test_search_node_uses_optimized_query_when_available() -> None:
    services = _OptimizingServices()
    state = {
        "last_decision": _DummyDecision(),
        "topic": "topic",
        "search_enabled": True,
        "search_status_message": None,
        "search_results": [],
        "markdown_path": "logs/debate_test.md",
        "turn_count": 0,
        "continuation_mode": False,
        "transcript": [],
    }

    with patch("app.nodes.search.optimize_search_query", return_value="optimized query"):
        updated = search_node(state, services)

    assert services.search_service.last_query == "optimized query"
    assert updated["search_results"][0].query == "optimized query"
