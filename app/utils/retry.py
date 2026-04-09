"""Retry policies separated by failure mode."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from app.llm.ollama_client import (
    OllamaModelNotLoadedError,
    OllamaTimeoutError,
    StructuredOutputValidationError,
)
from app.services.search_service import SearchCLIError, SearchTimeoutError

T = TypeVar("T")


@dataclass(slots=True)
class LLMRetryPolicy:
    """Policy for LLM failure-specific retries."""

    structured_retries: int = 2
    timeout_retries: int = 2
    model_not_loaded_retries: int = 1
    base_backoff_seconds: float = 0.4


@dataclass(slots=True)
class SearchRetryPolicy:
    """Policy for search CLI failure-specific retries."""

    timeout_retries: int = 1
    cli_retries: int = 1
    base_backoff_seconds: float = 0.3


def run_with_llm_retry(
    operation: Callable[[], T],
    policy: LLMRetryPolicy,
    on_model_not_loaded: Callable[[], None] | None = None,
) -> T:
    """Run operation with differentiated retries for LLM error classes."""
    structured_count = 0
    timeout_count = 0
    model_count = 0

    while True:
        try:
            return operation()
        except StructuredOutputValidationError:
            if structured_count >= policy.structured_retries:
                raise
            structured_count += 1
            time.sleep(policy.base_backoff_seconds * (2**structured_count))
        except OllamaTimeoutError:
            if timeout_count >= policy.timeout_retries:
                raise
            timeout_count += 1
            time.sleep(policy.base_backoff_seconds * (2**timeout_count))
        except OllamaModelNotLoadedError:
            if model_count >= policy.model_not_loaded_retries:
                raise
            model_count += 1
            if on_model_not_loaded is not None:
                on_model_not_loaded()
            time.sleep(policy.base_backoff_seconds * (1.5**model_count))


def run_with_search_retry(operation: Callable[[], T], policy: SearchRetryPolicy) -> T:
    """Run operation with differentiated retries for search failure classes."""
    timeout_count = 0
    cli_count = 0

    while True:
        try:
            return operation()
        except SearchTimeoutError:
            if timeout_count >= policy.timeout_retries:
                raise
            timeout_count += 1
            time.sleep(policy.base_backoff_seconds * (2**timeout_count))
        except SearchCLIError:
            if cli_count >= policy.cli_retries:
                raise
            cli_count += 1
            time.sleep(policy.base_backoff_seconds * (1.5**cli_count))
