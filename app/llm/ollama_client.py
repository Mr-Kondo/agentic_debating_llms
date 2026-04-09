"""Ollama API client for structured and plain generations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

TModel = TypeVar("TModel", bound=BaseModel)


class OllamaClientError(RuntimeError):
    """Base exception for Ollama client failures."""


class OllamaTimeoutError(OllamaClientError):
    """Raised when Ollama request times out."""


class OllamaModelNotLoadedError(OllamaClientError):
    """Raised when requested model appears unavailable."""


class StructuredOutputValidationError(OllamaClientError):
    """Raised when structured output cannot be parsed/validated."""


@dataclass(slots=True)
class OllamaClient:
    """Lightweight wrapper around Ollama HTTP endpoints."""

    base_url: str
    timeout_seconds: int = 60

    @staticmethod
    def _looks_like_model_not_found(message: str) -> bool:
        lowered = message.lower()
        model_hint = "model" in lowered
        missing_hint = (
            "not found" in lowered or "not loaded" in lowered or "no such model" in lowered or "unknown model" in lowered
        )
        return model_hint and missing_hint

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        try:
            response = httpx.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(f"Timeout calling Ollama at {url}") from exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            msg = exc.response.text
            if status_code == 404 and self._looks_like_model_not_found(msg):
                raise OllamaModelNotLoadedError(msg) from exc
            if status_code == 404:
                raise OllamaClientError(
                    f"Ollama endpoint not found at {exc.request.url}. "
                    f"Check OLLAMA_BASE_URL and available API endpoints. Response: {msg}"
                ) from exc
            raise OllamaClientError(f"Ollama HTTP {status_code} error: {msg}") from exc
        except httpx.HTTPError as exc:
            raise OllamaClientError(f"Ollama transport error: {exc}") from exc
        except ValueError as exc:
            raise OllamaClientError("Ollama returned non-JSON response") from exc

        if "error" in data and data["error"]:
            msg = str(data["error"])
            if self._looks_like_model_not_found(msg):
                raise OllamaModelNotLoadedError(msg)
            raise OllamaClientError(msg)

        return data

    @staticmethod
    def _extract_text_response(data: dict[str, Any]) -> str:
        response_text = data.get("response")
        if not isinstance(response_text, str):
            raise StructuredOutputValidationError("Ollama response does not include text in 'response'.")
        return response_text.strip()

    def generate_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[TModel],
        keep_alive: str | None = None,
    ) -> TModel:
        """Call /api/generate with JSON schema and validate output."""
        payload: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "format": schema_model.model_json_schema(),
        }
        if keep_alive:
            payload["keep_alive"] = keep_alive

        data = self._request("/api/generate", payload)
        raw_text = self._extract_text_response(data)

        try:
            json_payload = json.loads(raw_text)
            return schema_model.model_validate(json_payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise StructuredOutputValidationError(
                f"Failed to parse structured output for model {model}: {raw_text[:300]}"
            ) from exc

    def generate_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        keep_alive: str | None = None,
    ) -> str:
        """Call /api/generate and return plain text response."""
        payload: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
        }
        if keep_alive:
            payload["keep_alive"] = keep_alive

        data = self._request("/api/generate", payload)
        return self._extract_text_response(data)

    def list_loaded_models(self) -> list[str]:
        """Return model names reported by /api/ps."""
        data = self._request("/api/ps", {})
        models = data.get("models", [])
        names: list[str] = []
        for model_info in models:
            if isinstance(model_info, dict) and isinstance(model_info.get("name"), str):
                names.append(model_info["name"])
        return names
