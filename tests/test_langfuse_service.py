"""Tests for LangfuseService usage propagation in log_generation."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.langfuse_service import LangfuseService


def _make_disabled_service() -> LangfuseService:
    return LangfuseService(enabled=False, host="", public_key="", secret_key="")


def _make_enabled_service() -> LangfuseService:
    svc = LangfuseService.__new__(LangfuseService)
    object.__setattr__(svc, "enabled", True)
    object.__setattr__(svc, "host", "http://localhost")
    object.__setattr__(svc, "public_key", "pk")
    object.__setattr__(svc, "secret_key", "sk")
    object.__setattr__(svc, "_client", MagicMock())
    object.__setattr__(svc, "_trace_id", "trace-123")
    return svc


class TestLogGenerationWithUsage:
    def test_disabled_service_is_noop(self):
        """log_generation on a disabled service must not raise."""
        svc = _make_disabled_service()
        svc.log_generation(
            span=None,
            model="test",
            prompt="hello",
            completion="world",
            usage_details={"prompt_tokens": 10, "completion_tokens": 5},
        )  # should not raise

    def test_usage_passed_to_start_observation(self):
        """When usage is provided it is forwarded to the Langfuse client."""
        svc = _make_enabled_service()
        mock_gen = MagicMock()
        svc._client.start_observation.return_value = mock_gen

        usage = {"prompt_tokens": 12, "completion_tokens": 34}
        svc.log_generation(
            span=None,
            model="llama3.1",
            prompt="p",
            completion="c",
            usage_details=usage,
        )

        call_kwargs = svc._client.start_observation.call_args.kwargs
        assert call_kwargs["usage_details"] == usage

    def test_no_usage_does_not_pass_usage_key(self):
        """When usage is None the 'usage' key must not appear in kwargs."""
        svc = _make_enabled_service()
        mock_gen = MagicMock()
        svc._client.start_observation.return_value = mock_gen

        svc.log_generation(
            span=None,
            model="llama3.1",
            prompt="p",
            completion="c",
            usage_details=None,
        )

        call_kwargs = svc._client.start_observation.call_args.kwargs
        assert "usage_details" not in call_kwargs

    def test_generation_end_called(self):
        """The generation object's end() is always called after logging."""
        svc = _make_enabled_service()
        mock_gen = MagicMock()
        svc._client.start_observation.return_value = mock_gen

        svc.log_generation(span=None, model="m", prompt="p", completion="c")

        mock_gen.end.assert_called_once()
