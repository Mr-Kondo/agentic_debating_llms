"""Langfuse tracing wrapper with graceful disable behavior."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(slots=True)
class LangfuseService:
    """Thin wrapper for Langfuse trace/span/generation logging."""

    enabled: bool
    host: str
    public_key: str
    secret_key: str
    _client: Any | None = field(default=None, init=False, repr=False)
    _trace: Any | None = field(default=None, init=False, repr=False)

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
        except Exception:
            self.enabled = False

    def start_trace(self, session_id: str, topic: str) -> None:
        """Start a single trace for one discussion session."""
        if not self.enabled or self._client is None:
            return
        try:
            self._trace = self._client.trace(
                id=session_id,
                name="local-llm-debate-session",
                input={"topic": topic},
            )
        except Exception:
            self.enabled = False

    def end_trace(self, output: dict[str, Any] | None = None) -> None:
        """Finalize current trace."""
        if not self.enabled or self._trace is None:
            return
        try:
            if hasattr(self._trace, "update"):
                self._trace.update(output=output or {})
            if self._client is not None and hasattr(self._client, "flush"):
                self._client.flush()
        except Exception:
            return

    @contextmanager
    def span(self, name: str, input_data: dict[str, Any] | None = None) -> Iterator[Any]:
        """Create a span context if enabled, otherwise no-op."""
        span_obj: Any | None = None
        if self.enabled and self._trace is not None:
            try:
                span_obj = self._trace.span(name=name, input=input_data or {})
            except Exception:
                span_obj = None

        try:
            yield span_obj
            if span_obj is not None and hasattr(span_obj, "end"):
                span_obj.end()
        except Exception as exc:
            self.log_error(str(exc), error_type=type(exc).__name__, span=span_obj)
            if span_obj is not None and hasattr(span_obj, "end"):
                span_obj.end(output={"error": str(exc)})
            raise

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

        target = span if span is not None else self._trace
        if target is None:
            return

        try:
            if hasattr(target, "generation"):
                target.generation(
                    name="ollama-generation",
                    model=model,
                    input=prompt,
                    output=completion,
                    metadata=metadata or {},
                )
        except Exception:
            return

    def log_error(self, message: str, error_type: str = "RuntimeError", span: Any | None = None) -> None:
        """Record errors to Langfuse if enabled."""
        if not self.enabled:
            return

        target = span if span is not None else self._trace
        if target is None:
            return

        try:
            if hasattr(target, "event"):
                target.event(
                    name="error",
                    level="ERROR",
                    input={"error_type": error_type},
                    output={"message": message},
                )
        except Exception:
            return
