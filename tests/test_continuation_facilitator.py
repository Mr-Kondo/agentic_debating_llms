from __future__ import annotations

from contextlib import contextmanager

from app.nodes.continuation_facilitator import continuation_facilitator_node
from app.prompts import build_continuation_facilitator_prompt
from app.schemas import ContinuationDecision


class _DummyConfig:
    facilitator_model = "llama3.1:8b"
    model_keep_alive = "10m"
    recent_context_turns = 4


class _SuccessClient:
    def generate_structured(self, **kwargs):
        _ = kwargs
        return ContinuationDecision(
            action="continue_a",
            reason="Explore a blind spot",
            focus_instruction="Probe one missing assumption",
        )


class _FailingClient:
    def generate_structured(self, **kwargs):
        _ = kwargs
        raise RuntimeError("boom")


class _DummyModelManager:
    def ensure_loaded(self, model: str, warmup: bool = True) -> None:
        _ = (model, warmup)


class _DummyMarkdownLogger:
    def append_continuation_decision(self, path, decision) -> None:
        _ = (path, decision)


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
    def __init__(self, client) -> None:
        self.config = _DummyConfig()
        self.ollama_client = client
        self.model_manager = _DummyModelManager()
        self.markdown_logger = _DummyMarkdownLogger()
        self.langfuse = _DummyLangfuse()


def _base_state() -> dict:
    return {
        "topic": "test topic",
        "transcript": [],
        "search_results": [],
        "validation_log": [],
        "compact_summary": "summary",
        "turn_count": 8,
        "max_turns": 8,
        "next_action": "speak_a",
        "last_decision": {"action": "finish", "reason": "done"},
        "final_summary": "initial conclusion",
        "markdown_path": "/tmp/test.md",
        "result_markdown_path": None,
        "input_sources": [],
        "session_id": "test-sid",
        "last_error": None,
        "continuation_mode": True,
        "continuation_turn_count": 0,
        "continuation_max_turns": 3,
    }


def test_continuation_prompt_displays_human_friendly_round_number() -> None:
    prompt = build_continuation_facilitator_prompt(
        topic="test topic",
        final_summary="initial conclusion",
        compact_summary="summary",
        recent_turns=[],
        continuation_turn_count=0,
        continuation_max_turns=3,
    )

    assert "Continuation round: 1/3" in prompt


def test_continuation_facilitator_increments_counter_on_success() -> None:
    updated = continuation_facilitator_node(_base_state(), _DummyServices(_SuccessClient()))

    assert updated["continuation_turn_count"] == 1
    assert updated["next_action"] == "speak_a"


def test_continuation_facilitator_does_not_increment_counter_on_error() -> None:
    updated = continuation_facilitator_node(_base_state(), _DummyServices(_FailingClient()))

    assert updated["continuation_turn_count"] == 0
    assert updated["next_action"] == "conclude"
    assert updated["last_error"] == "continuation_facilitator_error:boom"
