"""Search service that invokes external CLI through subprocess."""

from __future__ import annotations

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

    def _build_command(self, query: str) -> list[str]:
        rendered = self.command_template.format(query=query)
        return shlex.split(rendered)

    def run(self, query: str) -> SearchResult:
        """Execute configured search CLI command and return SearchResult."""
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
            raise SearchCLIError("Search CLI exited with non-zero status.", result=result)

        return result
