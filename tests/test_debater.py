from __future__ import annotations

from contextlib import contextmanager

from app.nodes.debater import debater_a_node
from app.schemas import DebaterResponse


class _DummyConfig:
    debater_a_model = "gemma4:latest"
    debater_b_model = "qwen3.5:latest"
    model_keep_alive = "10m"
    recent_context_turns = 4


class _SearchRequestClient:
    def generate_structured(self, **kwargs):
        _ = kwargs
        return DebaterResponse(
            speaker="A",
            claim="Need data before concluding.",
            stance_summary="insufficient evidence",
            confidence=0.5,
            needs_search=True,
            search_query="latest benchmark comparison",
            search_reason="Need objective evidence",
        )


class _DummyModelManager:
    def ensure_loaded(self, model: str, warmup: bool = True) -> None:
        _ = (model, warmup)


class _DummyMarkdownLogger:
    def append_debater_utterance(self, path, response) -> None:
        _ = (path, response)


class _DummyLangfuse:
    @contextmanager
    def span(self, name: str, input_data=None):
        _ = (name, input_data)
        yield object()

    def log_generation(self, **kwargs) -> None:
        _ = kwargs


class _DummyServices:
    def __init__(self):
        self.config = _DummyConfig()
        self.ollama_client = _SearchRequestClient()
        self.model_manager = _DummyModelManager()
        self.markdown_logger = _DummyMarkdownLogger()
        self.langfuse = _DummyLangfuse()


def test_debater_node_can_request_search() -> None:
    state = {
        "topic": "topic",
        "transcript": [],
        "search_results": [],
        "search_enabled": True,
        "search_status_message": None,
        "validation_log": [],
        "compact_summary": "summary",
        "turn_count": 2,
        "max_turns": 8,
        "next_action": "speak_a",
        "last_decision": {
            "action": "speak_a",
            "reason": "test",
            "focus_instruction": "add evidence",
        },
        "final_summary": None,
        "markdown_path": "x.md",
        "result_markdown_path": None,
        "input_sources": [],
        "session_id": "sid",
        "last_error": None,
        "continuation_mode": False,
        "continuation_turn_count": 0,
        "continuation_max_turns": 0,
    }

    updated = debater_a_node(state, _DummyServices())

    assert updated["next_action"] == "search"
    assert updated["last_decision"]["search_query"] == "latest benchmark comparison"
    assert updated["last_decision"]["request_source"] == "debater_a"
