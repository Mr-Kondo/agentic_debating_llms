"""Tests for turn_count behavior in continuation mode."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

from app.nodes.debater import debater_a_node, debater_b_node
from app.nodes.search import search_node
from app.schemas import DebaterResponse, SearchResult


class _DummyConfig:
    facilitator_model = "llama3.1:8b"
    debater_a_model = "gemma4:8b"
    debater_b_model = "qwen3.5:8b"
    validator_model = "rnj-1:latest"
    model_keep_alive = "10m"
    recent_context_turns = 4
    search_command_template = 'echo "{query}"'
    search_timeout_seconds = 5


class _DummyClient:
    def generate_structured(self, **kwargs):
        schema = kwargs.get("schema_model")
        if schema is DebaterResponse:
            return DebaterResponse(
                speaker="A",
                claim="test claim",
                stance_summary="test stance",
                confidence=0.8,
            )
        raise ValueError(f"Unexpected schema: {schema}")


class _DummyModelManager:
    def ensure_loaded(self, model: str, warmup: bool = True) -> None:
        pass


class _DummyMarkdownLogger:
    def append_debater_utterance(self, path, response) -> None:
        pass

    def append_search_result(self, path, result) -> None:
        pass


class _DummyLangfuse:
    @contextmanager
    def span(self, name: str, input_data=None):
        yield object()

    def log_generation(self, **kwargs) -> None:
        pass

    def log_error(self, *args, **kwargs) -> None:
        pass


class _DummySearchService:
    def run(self, query: str) -> SearchResult:
        return SearchResult(
            query=query,
            stdout="result",
            stderr="",
            returncode=0,
            digest="test digest",
        )


class _DummyServices:
    def __init__(self):
        self.config = _DummyConfig()
        self.ollama_client = _DummyClient()
        self.model_manager = _DummyModelManager()
        self.markdown_logger = _DummyMarkdownLogger()
        self.langfuse = _DummyLangfuse()
        self.search_service = _DummySearchService()


def _base_state(*, continuation_mode: bool = False) -> dict:
    return {
        "topic": "test topic",
        "transcript": [],
        "search_results": [],
        "validation_log": [],
        "compact_summary": "summary",
        "turn_count": 8,
        "max_turns": 8,
        "next_action": "speak_a",
        "last_decision": {
            "action": "speak_a",
            "reason": "test",
            "focus_instruction": "test focus",
            "search_query": "test query",
        },
        "final_summary": "initial conclusion",
        "markdown_path": "/tmp/test.md",
        "result_markdown_path": None,
        "input_sources": [],
        "session_id": "test-sid",
        "last_error": None,
        "continuation_mode": continuation_mode,
        "continuation_turn_count": 0,
        "continuation_max_turns": 3,
    }


# --- Debater turn_count tests ---


def test_debater_increments_turn_count_in_normal_mode() -> None:
    state = _base_state(continuation_mode=False)
    state["turn_count"] = 3
    state["max_turns"] = 8
    result = debater_a_node(state, _DummyServices())
    assert result["turn_count"] == 4


def test_debater_does_not_increment_turn_count_in_continuation_mode() -> None:
    state = _base_state(continuation_mode=True)
    result = debater_a_node(state, _DummyServices())
    assert result["turn_count"] == 8


def test_debater_b_does_not_increment_turn_count_in_continuation_mode() -> None:
    state = _base_state(continuation_mode=True)
    state["last_decision"]["action"] = "speak_b"
    state["next_action"] = "speak_b"
    result = debater_b_node(state, _DummyServices())
    assert result["turn_count"] == 8


# --- Search turn_count tests ---


def test_search_increments_turn_count_in_normal_mode() -> None:
    state = _base_state(continuation_mode=False)
    state["turn_count"] = 3
    state["max_turns"] = 8
    result = search_node(state, _DummyServices())
    assert result["turn_count"] == 4


def test_search_does_not_increment_turn_count_in_continuation_mode() -> None:
    state = _base_state(continuation_mode=True)
    result = search_node(state, _DummyServices())
    assert result["turn_count"] == 8
