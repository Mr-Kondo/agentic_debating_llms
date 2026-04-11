"""Search service that invokes external CLI through subprocess."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass

from app.llm.interfaces import SearchDigestProvider
from app.schemas import SearchResult


class SearchServiceError(RuntimeError):
    """Base class for search service failures."""


class SearchCLIError(SearchServiceError):
    """Raised when search CLI exits with non-zero return code."""

    def __init__(self, message: str, result: SearchResult):
        super().__init__(message)
        self.result = result


class SearchTimeoutError(SearchServiceError):
    """Raised when search CLI call times out."""

    def __init__(self, message: str, query: str):
        super().__init__(message)
        self.query = query


@dataclass(slots=True)
class DefaultSearchDigester(SearchDigestProvider):
    """Simple bounded digest builder for search outputs."""

    max_chars: int = 800

    def digest(self, stdout: str, stderr: str, returncode: int) -> str:
        stdout_clean = " ".join(stdout.strip().split())
        stderr_clean = " ".join(stderr.strip().split())
        summary = f"returncode={returncode}; stdout={stdout_clean}; stderr={stderr_clean}".strip()
        if len(summary) > self.max_chars:
            return summary[: self.max_chars - 3] + "..."
        return summary


@dataclass(slots=True)
class SearchService:
    """Execute external search command and collect raw + digested output."""

    command_template: str
    timeout_seconds: int
    digester: SearchDigestProvider
    backend: str = "api"
    max_results: int = 5

    def _build_command(self, query: str) -> list[str]:
        rendered = self.command_template.format(query=query)
        return shlex.split(rendered)

    @staticmethod
    def _classify_nonzero(returncode: int, stderr: str) -> str:
        stderr_lower = stderr.lower()
        if returncode == 127:
            return "cli_not_found"
        if returncode in (2, 64):
            return "invalid_template_or_args"
        if "missing option" in stderr_lower or "no such command" in stderr_lower or "invalid value" in stderr_lower:
            return "invalid_template_or_args"
        return "runtime_failure"

    @staticmethod
    def _validate_query(query: str) -> str | None:
        if not query.strip():
            return "query must not be empty"
        if "\n" in query or "\r" in query or "\x00" in query:
            return "query must not include control characters (newline/NULL)"
        return None

    def run(self, query: str) -> SearchResult:
        """Execute configured search CLI command and return SearchResult."""
        query_error = self._validate_query(query)
        if query_error is not None:
            result = SearchResult(
                query=query,
                stdout="",
                stderr=f"Invalid search query: {query_error}",
                returncode=2,
                digest=self.digester.digest("", f"Invalid search query: {query_error}", 2),
            )
            raise SearchCLIError(
                "Search query rejected (invalid_template_or_args). Remove control characters and ensure query is non-empty.",
                result=result,
            )

        if self.backend == "api":
            return self._run_api(query)
        return self._run_cli(query)

    def _run_api(self, query: str) -> SearchResult:
        """Execute search via ddgs Python API."""
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS  # pragma: no cover - compatibility path
        except Exception as exc:
            result = SearchResult(
                query=query,
                stdout="",
                stderr=f"Search API client not available: {exc}",
                returncode=127,
                digest=self.digester.digest("", f"Search API client not available: {exc}", 127),
            )
            raise SearchCLIError("Search API backend is unavailable.", result=result) from exc

        try:
            ddgs = DDGS()
            try:
                raw_results = ddgs.text(query=query, max_results=self.max_results)
            except TypeError:
                # Compatibility for older client signatures.
                raw_results = ddgs.text(keywords=query, max_results=self.max_results)
            results = list(raw_results or [])
            stdout = json.dumps(results, ensure_ascii=False)
            result = SearchResult(
                query=query,
                stdout=stdout,
                stderr="",
                returncode=0,
                digest=self.digester.digest(stdout, "", 0),
            )
            return result
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if "timeout" in lowered:
                raise SearchTimeoutError(f"Search API timed out for query: {query}", query=query) from exc

            result = SearchResult(
                query=query,
                stdout="",
                stderr=f"Search API error: {type(exc).__name__}: {message}",
                returncode=1,
                digest=self.digester.digest("", f"Search API error: {type(exc).__name__}: {message}", 1),
            )
            raise SearchCLIError("Search API failed (runtime_failure).", result=result) from exc

    def _run_cli(self, query: str) -> SearchResult:
        """Execute search via external CLI command."""

        command = self._build_command(query)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SearchTimeoutError(f"Search command timed out for query: {query}", query=query) from exc
        except FileNotFoundError as exc:
            result = SearchResult(
                query=query,
                stdout="",
                stderr=f"Search CLI not found: {command[0]}",
                returncode=127,
                digest=self.digester.digest("", f"Search CLI not found: {command[0]}", 127),
            )
            raise SearchCLIError("Search CLI binary not found.", result=result) from exc

        result = SearchResult(
            query=query,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            digest=self.digester.digest(completed.stdout, completed.stderr, completed.returncode),
        )

        if completed.returncode != 0:
            category = self._classify_nonzero(completed.returncode, completed.stderr)
            raise SearchCLIError(
                "Search CLI exited with non-zero status "
                f"({category}). Check SEARCH_COMMAND_TEMPLATE and ddgs '-q/--query' usage.",
                result=result,
            )

        return result
