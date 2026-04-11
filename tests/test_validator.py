from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

from app.nodes.validator import validator_node
from app.schemas import ValidatorFeedback


class _DummyConfig:
    validator_model = "rnj-1:latest"
    model_keep_alive = "10m"
    recent_context_turns = 4


class _DummyClient:
    def generate_structured(self, **kwargs):
        return ValidatorFeedback(
            is_valid=True,
            confidence=0.9,
            issues="none",
            improvement="continue",
        )


class _SearchRequestClient:
    def generate_structured(self, **kwargs):
        return ValidatorFeedback(
            is_valid=False,
            confidence=0.4,
            issues="claim lacks evidence",
            improvement="add external references",
            needs_search=True,
            search_query="recent evidence on the topic",
            search_reason="Need factual verification",
        )


class _DummyModelManager:
    def ensure_loaded(self, model: str, warmup: bool = True) -> None:
        _ = (model, warmup)


class _DummyMarkdownLogger:
    def __init__(self) -> None:
        self.called = False

    def append_validator_feedback(self, path, feedback) -> None:
        _ = (path, feedback)
        self.called = True


class _DummyLangfuse:
    @contextmanager
    def span(self, name: str, input_data=None):
        _ = (name, input_data)
        yield object()

    def log_generation(self, **kwargs) -> None:
        _ = kwargs

    def log_error(self, *args, **kwargs) -> None:
        _ = (args, kwargs)


class _DummyServices:
    def __init__(self):
        self.config = _DummyConfig()
        self.ollama_client = _DummyClient()
        self.model_manager = _DummyModelManager()
        self.markdown_logger = _DummyMarkdownLogger()
        self.langfuse = _DummyLangfuse()


def test_validator_node_appends_feedback() -> None:
    state = {
        "topic": "topic",
        "transcript": [
            {
                "role": "Debater A",
                "content": "claim text",
                "timestamp": datetime.now(timezone.utc),
            }
        ],
        "search_results": [],
        "validation_log": [],
        "compact_summary": "summary",
        "turn_count": 1,
        "max_turns": 8,
        "next_action": "speak_b",
        "last_decision": {"action": "speak_b", "reason": "x"},
        "final_summary": None,
        "markdown_path": "x.md",
        "result_markdown_path": None,
        "input_sources": [],
        "session_id": "sid",
        "last_error": None,
    }

    services = _DummyServices()
    updated = validator_node(state, services)

    assert "validation_log" in updated
    assert len(updated["validation_log"]) == 1
    assert services.markdown_logger.called is True


def test_validator_node_can_request_search() -> None:
    state = {
        "topic": "topic",
        "transcript": [
            {
                "role": "Debater A",
                "content": "claim text",
                "timestamp": datetime.now(timezone.utc),
            }
        ],
        "search_results": [],
        "search_enabled": True,
        "search_status_message": None,
        "validation_log": [],
        "compact_summary": "summary",
        "turn_count": 1,
        "max_turns": 8,
        "next_action": "speak_b",
        "last_decision": {"action": "speak_b", "reason": "x"},
        "final_summary": None,
        "markdown_path": "x.md",
        "result_markdown_path": None,
        "input_sources": [],
        "session_id": "sid",
        "last_error": None,
    }

    services = _DummyServices()
    services.ollama_client = _SearchRequestClient()
    updated = validator_node(state, services)

    assert updated["next_action"] == "search"
    assert updated["last_decision"]["search_query"] == "recent evidence on the topic"
