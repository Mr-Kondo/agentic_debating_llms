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
    def _strip_think_tags(text: str) -> str:
        """Remove only leading <think>...</think> reasoning blocks.

        This intentionally strips balanced think blocks only at the beginning of
        the model response so literal ``<think>`` strings inside the actual JSON
        or text content are preserved.
        """
        open_tag = "<think>"
        close_tag = "</think>"
        length = len(text)
        index = 0

        while index < length and text[index].isspace():
            index += 1

        removed_any = False

        while text.startswith(open_tag, index):
            removed_any = True
            block_start = index
            depth = 0

            while index < length:
                if text.startswith(open_tag, index):
                    depth += 1
                    index += len(open_tag)
                    continue
                if text.startswith(close_tag, index):
                    depth -= 1
                    index += len(close_tag)
                    if depth == 0:
                        while index < length and text[index].isspace():
                            index += 1
                        break
                    continue
                index += 1
            else:
                return text[block_start:].strip() if removed_any else text.strip()

        if not removed_any:
            return text.strip()

        return text[index:].strip()

    @staticmethod
    def _extract_text_response(data: dict[str, Any]) -> str:
        response_text = data.get("response")
        if isinstance(response_text, str) and response_text.strip():
            return response_text.strip()

        for key in ("output", "content", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        message_obj = data.get("message")
        if isinstance(message_obj, dict):
            value = message_obj.get("content")
            if isinstance(value, str) and value.strip():
                return value.strip()

        raise StructuredOutputValidationError(
            "Ollama response did not include usable text. "
            "response was empty or missing; this can happen when model thinking output is enabled."
        )

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Try to extract a JSON object from text that may contain prose or code fences.

        Handles:
        - ```json ... ``` and ``` ... ``` code fences
        - Text before/after a bare ``{...}`` object
        Returns the extracted JSON string, or None if no object found.
        """
        import re

        # Strip code fences first
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence_match:
            return fence_match.group(1).strip()

        # Find the outermost {...} by depth traversal
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

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
            "think": False,
            "options": {"thinking": False},
        }
        if keep_alive:
            payload["keep_alive"] = keep_alive

        data = self._request("/api/generate", payload)
        raw_text = self._extract_text_response(data)

        # Attempt 1: direct parse
        try:
            json_payload = json.loads(raw_text)
            return schema_model.model_validate(json_payload)
        except (json.JSONDecodeError, ValidationError):
            pass

        # Attempt 2: strip leading <think>...</think> blocks
        candidate = self._strip_think_tags(raw_text)
        try:
            json_payload = json.loads(candidate)
            return schema_model.model_validate(json_payload)
        except (json.JSONDecodeError, ValidationError):
            pass

        # Attempt 3: extract embedded JSON object (handles code fences, surrounding prose)
        extracted = self._extract_json_object(candidate)
        if extracted is not None:
            try:
                json_payload = json.loads(extracted)
                return schema_model.model_validate(json_payload)
            except (json.JSONDecodeError, ValidationError):
                pass

        raise StructuredOutputValidationError(f"Failed to parse structured output for model {model}: {raw_text}")

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
            "think": False,
            "options": {"thinking": False},
        }
        if keep_alive:
            payload["keep_alive"] = keep_alive

        data = self._request("/api/generate", payload)
        return self._strip_think_tags(self._extract_text_response(data))

    def list_loaded_models(self) -> list[str]:
        """Return model names reported by /api/ps."""
        data = self._request("/api/ps", {})
        models = data.get("models", [])
        names: list[str] = []
        for model_info in models:
            if isinstance(model_info, dict) and isinstance(model_info.get("name"), str):
                names.append(model_info["name"])
        return names
