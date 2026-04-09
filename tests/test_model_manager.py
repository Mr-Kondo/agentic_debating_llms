from __future__ import annotations

from app.llm.model_manager import ModelManager
from app.llm.ollama_client import OllamaClientError, OllamaModelNotLoadedError, OllamaTimeoutError


class _FakeClientSuccess:
    def list_loaded_models(self) -> list[str]:
        return []

    def generate_text(self, **kwargs) -> str:
        return "ok"


class _FakeClientMissing:
    def list_loaded_models(self) -> list[str]:
        return []

    def generate_text(self, **kwargs) -> str:
        raise OllamaModelNotLoadedError("model not found")


class _FakeClientTimeout:
    def list_loaded_models(self) -> list[str]:
        return []

    def generate_text(self, **kwargs) -> str:
        raise OllamaTimeoutError("timeout")


class _FakeClientClientError:
    def list_loaded_models(self) -> list[str]:
        return []

    def generate_text(self, **kwargs) -> str:
        raise OllamaClientError("endpoint not found")


def test_preload_models_returns_success_for_loaded_models() -> None:
    manager = ModelManager(client=_FakeClientSuccess(), keep_alive="10m")
    result = manager.preload_models(["a:1", "b:1"], warmup=True)
    assert result == {"a:1": "success", "b:1": "success"}


def test_preload_models_returns_not_found() -> None:
    manager = ModelManager(client=_FakeClientMissing(), keep_alive="10m")
    result = manager.preload_models(["missing:1"], warmup=False)
    assert result == {"missing:1": "not_found"}


def test_preload_models_returns_timeout_error() -> None:
    manager = ModelManager(client=_FakeClientTimeout(), keep_alive="10m")
    result = manager.preload_models(["slow:1"], warmup=False)
    assert result == {"slow:1": "error:timeout"}


def test_preload_models_returns_client_error() -> None:
    manager = ModelManager(client=_FakeClientClientError(), keep_alive="10m")
    result = manager.preload_models(["bad:1"], warmup=False)
    assert result == {"bad:1": "error:client"}
