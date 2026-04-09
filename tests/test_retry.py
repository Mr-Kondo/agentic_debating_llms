from __future__ import annotations

import pytest

from app.llm.ollama_client import (
    OllamaModelNotLoadedError,
    OllamaTimeoutError,
    StructuredOutputValidationError,
)
from app.services.search_service import SearchCLIError, SearchResult, SearchTimeoutError
from app.utils.retry import LLMRetryPolicy, SearchRetryPolicy, run_with_llm_retry, run_with_search_retry


def test_llm_retry_structured_validation_then_success() -> None:
    calls = {"count": 0}

    def operation() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise StructuredOutputValidationError("bad json")
        return "ok"

    result = run_with_llm_retry(operation, policy=LLMRetryPolicy(structured_retries=2, timeout_retries=0))
    assert result == "ok"
    assert calls["count"] == 3


def test_llm_retry_timeout_exhausted() -> None:
    def operation() -> str:
        raise OllamaTimeoutError("timeout")

    with pytest.raises(OllamaTimeoutError):
        run_with_llm_retry(operation, policy=LLMRetryPolicy(structured_retries=0, timeout_retries=1))


def test_llm_retry_model_not_loaded_recovery_called() -> None:
    calls = {"count": 0, "recover": 0}

    def operation() -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise OllamaModelNotLoadedError("not loaded")
        return "ok"

    def recover() -> None:
        calls["recover"] += 1

    result = run_with_llm_retry(
        operation,
        policy=LLMRetryPolicy(model_not_loaded_retries=1, structured_retries=0, timeout_retries=0),
        on_model_not_loaded=recover,
    )
    assert result == "ok"
    assert calls["recover"] == 1


def test_search_retry_cli_failure_then_success() -> None:
    calls = {"count": 0}

    def operation() -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise SearchCLIError(
                "failed",
                result=SearchResult(query="q", stdout="", stderr="err", returncode=1, digest="d"),
            )
        return "ok"

    result = run_with_search_retry(operation, policy=SearchRetryPolicy(cli_retries=1, timeout_retries=0))
    assert result == "ok"


def test_search_retry_timeout_exhausted() -> None:
    def operation() -> str:
        raise SearchTimeoutError("timeout", query="q")

    with pytest.raises(SearchTimeoutError):
        run_with_search_retry(operation, policy=SearchRetryPolicy(timeout_retries=1, cli_retries=0))
