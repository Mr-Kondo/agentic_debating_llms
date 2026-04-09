"""Model lifecycle management helpers for Ollama."""

from __future__ import annotations

from dataclasses import dataclass

from app.llm.ollama_client import OllamaClient, OllamaModelNotLoadedError


@dataclass(slots=True)
class ModelManager:
    """Manage model preload, warmup, unload, and process checks."""

    client: OllamaClient
    keep_alive: str

    def ps(self) -> list[str]:
        """List loaded models from Ollama."""
        return self.client.list_loaded_models()

    def is_loaded(self, model: str) -> bool:
        """Check whether the target model is currently loaded."""
        loaded = self.ps()
        return any(name.startswith(model) or model.startswith(name) for name in loaded)

    def preload(self, model: str) -> None:
        """Preload a model by issuing a minimal generation call."""
        self.client.generate_text(
            model=model,
            system_prompt="You are warming up.",
            user_prompt="warmup",
            keep_alive=self.keep_alive,
        )

    def warmup(self, model: str) -> None:
        """Warm up a model with a tiny deterministic prompt."""
        self.client.generate_text(
            model=model,
            system_prompt="Return OK.",
            user_prompt="Respond with OK only.",
            keep_alive=self.keep_alive,
        )

    def unload(self, model: str) -> None:
        """Request model unload by setting keep_alive to 0."""
        try:
            self.client.generate_text(
                model=model,
                system_prompt="Unload.",
                user_prompt="",
                keep_alive="0",
            )
        except OllamaModelNotLoadedError:
            return

    def preload_models(self, models: list[str], warmup: bool = True) -> None:
        """Preload and optionally warmup all models used by a session."""
        for model in models:
            self.preload(model)
            if warmup:
                self.warmup(model)

    def ensure_loaded(self, model: str, warmup: bool = True) -> None:
        """Ensure model is available, attempting preload if not loaded."""
        if self.is_loaded(model):
            return
        self.preload(model)
        if warmup:
            self.warmup(model)
