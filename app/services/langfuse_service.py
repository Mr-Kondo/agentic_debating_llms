"""Langfuse tracing wrapper with graceful disable behavior."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import sys
from typing import Any, Iterator


@dataclass(slots=True)
class LangfuseService:
    """Thin wrapper for Langfuse trace/span/generation logging."""

    enabled: bool
    host: str
    public_key: str
    secret_key: str
    _client: Any | None = field(default=None, init=False, repr=False)
    _trace_id: str | None = field(default=None, init=False, repr=False)

    @staticmethod
    def _warn(message: str) -> None:
        print(f"[langfuse] {message}", file=sys.stderr)

    def __post_init__(self) -> None:
        if not self.enabled:
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                host=self.host or None,
                public_key=self.public_key or None,
                secret_key=self.secret_key or None,
            )
        except Exception as exc:
            self._warn(f"initialization failed: {type(exc).__name__}: {exc}")
            self.enabled = False

    def startup_check(self) -> bool:
        """Validate Langfuse client availability on startup."""
        if not self.enabled:
            return False
        if self._client is None:
            self._warn("disabled because client is unavailable")
            self.enabled = False
            return False
        try:
            if hasattr(self._client, "auth_check"):
                self._client.auth_check()
            if hasattr(self._client, "flush"):
                self._client.flush()
            return True
        except Exception as exc:
            self._warn(f"startup check failed: {type(exc).__name__}: {exc}")
            self.enabled = False
            self._client = None
            return False

    def start_trace(self, session_id: str, topic: str) -> None:
        """Start a single trace for one discussion session."""
        if not self.enabled or self._client is None:
            return
        self._trace_id = session_id
        try:
            if hasattr(self._client, "create_event"):
                self._client.create_event(
                    trace_context={"trace_id": session_id},
                    name="local-llm-debate-session",
                    input={"topic": topic},
                    metadata={"kind": "session_start"},
                )
        except Exception as exc:
            self._warn(f"start_trace failed: {type(exc).__name__}: {exc}")
            self.enabled = False

    def end_trace(self, output: dict[str, Any] | None = None) -> None:
        """Finalize current trace."""
        if not self.enabled or self._trace_id is None:
            return
        try:
            if hasattr(self._client, "create_event"):
                self._client.create_event(
                    trace_context={"trace_id": self._trace_id},
                    name="session_end",
                    output=output or {},
                    metadata={"kind": "session_end"},
                )
            if self._client is not None and hasattr(self._client, "flush"):
                self._client.flush()
        except Exception as exc:
            self._warn(f"end_trace failed: {type(exc).__name__}: {exc}")
        finally:
            self._trace_id = None

    @contextmanager
    def span(self, name: str, input_data: dict[str, Any] | None = None) -> Iterator[Any]:
        """Create a span context if enabled, otherwise no-op."""
        if not self.enabled or self._client is None or self._trace_id is None:
            yield None
            return

        try:
            observation_cm = self._client.start_as_current_observation(
                trace_context={"trace_id": self._trace_id},
                name=name,
                as_type="span",
                input=input_data or {},
                end_on_exit=True,
            )
        except Exception as exc:
            self._warn(f"span start failed ({name}): {type(exc).__name__}: {exc}")
            self.log_error(str(exc), error_type=type(exc).__name__)
            yield None
            return

        with observation_cm as span_obj:
            yield span_obj

    def log_generation(
        self,
        *,
        span: Any,
        model: str,
        prompt: str,
        completion: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record LLM generation under a span when possible."""
        if not self.enabled:
            return

        if self._client is None or self._trace_id is None:
            return

        try:
            if hasattr(self._client, "start_observation"):
                generation = self._client.start_observation(
                    trace_context={"trace_id": self._trace_id},
                    name="ollama-generation",
                    as_type="generation",
                    model=model,
                    input=prompt,
                    output=completion,
                    metadata=metadata or {},
                )
                if hasattr(generation, "end"):
                    generation.end()
        except Exception as exc:
            self._warn(f"generation logging failed: {type(exc).__name__}: {exc}")

    def log_error(self, message: str, error_type: str = "RuntimeError", span: Any | None = None) -> None:
        """Record errors to Langfuse if enabled."""
        if not self.enabled:
            return

        if self._client is None or self._trace_id is None:
            return

        try:
            if hasattr(self._client, "create_event"):
                self._client.create_event(
                    trace_context={"trace_id": self._trace_id},
                    name="error",
                    level="ERROR",
                    input={"error_type": error_type},
                    output={"message": message},
                )
        except Exception as exc:
            self._warn(f"error event logging failed: {type(exc).__name__}: {exc}")
