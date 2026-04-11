from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from app.services.search_service import DefaultSearchDigester, SearchCLIError, SearchService


def test_search_service_missing_cli_returns_127_result() -> None:
    service = SearchService(
        command_template='ddgs text -q "{query}" --max-results 5',
        timeout_seconds=5,
        digester=DefaultSearchDigester(),
        backend="cli",
    )

    with patch("subprocess.run", side_effect=FileNotFoundError("ddgs not found")):
        with pytest.raises(SearchCLIError) as exc_info:
            service.run("arknights")

    result = exc_info.value.result
    assert result.returncode == 127
    assert "Search CLI not found: ddgs" in result.stderr
    assert "returncode=127" in result.digest


def test_search_service_invalid_template_or_args_classified() -> None:
    service = SearchService(
        command_template='ddgs text "{query}" --max-results 5',
        timeout_seconds=5,
        digester=DefaultSearchDigester(),
        backend="cli",
    )
    completed = subprocess.CompletedProcess(
        args=["ddgs", "text", "query"],
        returncode=2,
        stdout="",
        stderr="Error: Missing option '-q'",
    )

    with patch("subprocess.run", return_value=completed):
        with pytest.raises(SearchCLIError) as exc_info:
            service.run("arknights")

    assert "invalid_template_or_args" in str(exc_info.value)
    assert exc_info.value.result.returncode == 2


def test_search_service_rejects_query_with_newline() -> None:
    service = SearchService(
        command_template='ddgs text -q "{query}" --max-results 5',
        timeout_seconds=5,
        digester=DefaultSearchDigester(),
        backend="cli",
    )

    with pytest.raises(SearchCLIError) as exc_info:
        service.run("line1\nline2")

    assert "invalid_template_or_args" in str(exc_info.value)
    assert exc_info.value.result.returncode == 2


def test_search_service_build_command_handles_japanese_query() -> None:
    service = SearchService(
        command_template='ddgs text -q "{query}" --max-results 5',
        timeout_seconds=5,
        digester=DefaultSearchDigester(),
        backend="cli",
    )

    command = service._build_command("アークナイツ gameplay mechanics")
    assert command[:3] == ["ddgs", "text", "-q"]
    assert command[3] == "アークナイツ gameplay mechanics"


def test_search_service_api_backend_success() -> None:
    class _FakeDDGS:
        def text(self, *, query: str, max_results: int):
            assert query == "arknights"
            assert max_results == 3
            return [{"title": "t", "href": "u", "body": "b"}]

    service = SearchService(
        command_template='ddgs text -q "{query}" --max-results 5',
        timeout_seconds=5,
        digester=DefaultSearchDigester(),
        backend="api",
        max_results=3,
    )

    with patch("ddgs.DDGS", return_value=_FakeDDGS()):
        result = service.run("arknights")

    assert result.returncode == 0
    assert "title" in result.stdout


def test_search_service_api_backend_failure_wrapped() -> None:
    class _FakeDDGS:
        def text(self, *, query: str, max_results: int):
            _ = (query, max_results)
            raise RuntimeError("network down")

    service = SearchService(
        command_template='ddgs text -q "{query}" --max-results 5',
        timeout_seconds=5,
        digester=DefaultSearchDigester(),
        backend="api",
    )

    with patch("ddgs.DDGS", return_value=_FakeDDGS()):
        with pytest.raises(SearchCLIError) as exc_info:
            service.run("arknights")

    assert "runtime_failure" in str(exc_info.value)
    assert exc_info.value.result.returncode == 1
