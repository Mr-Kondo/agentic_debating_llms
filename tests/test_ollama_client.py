"""Tests for OllamaClient focusing on think-tag stripping."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

from app.llm.ollama_client import OllamaClient, StructuredOutputValidationError


class _Simple(BaseModel):
    value: str


# ---------------------------------------------------------------------------
# _strip_think_tags
# ---------------------------------------------------------------------------


class TestStripThinkTags:
    def test_no_think_tags_unchanged(self):
        text = '{"value": "hello"}'
        assert OllamaClient._strip_think_tags(text) == text

    def test_single_line_think_tag_removed(self):
        text = "<think>reason here</think>actual content"
        assert OllamaClient._strip_think_tags(text) == "actual content"

    def test_multiline_think_tag_removed(self):
        text = '<think>\nline one\nline two\n</think>\n{"value": "ok"}'
        assert OllamaClient._strip_think_tags(text) == '{"value": "ok"}'

    def test_whitespace_stripped_after_removal(self):
        text = "<think>...</think>  \n  result"
        assert OllamaClient._strip_think_tags(text) == "result"

    def test_empty_string_unchanged(self):
        assert OllamaClient._strip_think_tags("") == ""

    def test_only_think_block_returns_empty(self):
        assert OllamaClient._strip_think_tags("<think>internal</think>") == ""

    def test_multiple_leading_think_blocks_removed(self):
        text = "<think>first</think>\n<think>second</think>\nend"
        assert OllamaClient._strip_think_tags(text) == "end"

    def test_non_leading_think_block_is_preserved(self):
        text = "<think>first</think>middle<think>second</think>end"
        assert OllamaClient._strip_think_tags(text) == "middle<think>second</think>end"

    def test_unmatched_opening_tag_is_left_unchanged(self):
        # Unbalanced tags are ambiguous, so preserve the original content.
        text = "<think>reasoning without closing tag"
        assert OllamaClient._strip_think_tags(text) == text

    def test_balanced_blocks_before_unmatched_block_stay_stripped(self):
        text = "<think>first</think>\n<think>unfinished"
        assert OllamaClient._strip_think_tags(text) == "<think>unfinished"

    def test_nested_tags_fully_removed(self):
        # Leading nested think blocks are removed without leaking inner content.
        text = "<think>outer <think>inner</think> more</think>actual"
        assert OllamaClient._strip_think_tags(text) == "actual"

    def test_stray_closing_tag_is_preserved(self):
        text = "some text</think>rest"
        assert OllamaClient._strip_think_tags(text) == text

    def test_literal_think_string_inside_json_is_preserved(self):
        text = '{"value": "Use <think> tags literally"}'
        assert OllamaClient._strip_think_tags(text) == text

    def test_literal_think_string_inside_text_is_preserved(self):
        text = "Usage: wrap reasoning in <think>...</think> tags"
        assert OllamaClient._strip_think_tags(text) == text


# ---------------------------------------------------------------------------
# generate_structured — think tags stripped before JSON parse
# ---------------------------------------------------------------------------


def _make_client() -> OllamaClient:
    return OllamaClient(base_url="http://localhost:11434")


def _mock_response(raw: str) -> dict:
    return {"response": raw, "model": "test"}


class TestGenerateStructured:
    def test_plain_json_parsed(self):
        client = _make_client()
        raw = '{"value": "hello"}'
        with patch.object(OllamaClient, "_request", return_value=_mock_response(raw)):
            result = client.generate_structured(
                model="test-model",
                system_prompt="sys",
                user_prompt="usr",
                schema_model=_Simple,
            )
        assert result.value == "hello"

    def test_think_prefix_stripped_before_parse(self):
        client = _make_client()
        raw = '<think>\nI will output JSON.\n</think>\n{"value": "world"}'
        with patch.object(OllamaClient, "_request", return_value=_mock_response(raw)):
            result = client.generate_structured(
                model="test-model",
                system_prompt="sys",
                user_prompt="usr",
                schema_model=_Simple,
            )
        assert result.value == "world"

    def test_literal_think_string_in_json_is_preserved(self):
        client = _make_client()
        raw = '{"value": "Explain <think> tags to users"}'
        with patch.object(OllamaClient, "_request", return_value=_mock_response(raw)):
            result = client.generate_structured(
                model="test-model",
                system_prompt="sys",
                user_prompt="usr",
                schema_model=_Simple,
            )
        assert result.value == "Explain <think> tags to users"

    def test_invalid_json_raises_structured_error(self):
        client = _make_client()
        raw = "not json at all"
        with patch.object(OllamaClient, "_request", return_value=_mock_response(raw)):
            with pytest.raises(StructuredOutputValidationError):
                client.generate_structured(
                    model="test-model",
                    system_prompt="sys",
                    user_prompt="usr",
                    schema_model=_Simple,
                )

    def test_schema_mismatch_raises_structured_error(self):
        # {"value": 123} fails Pydantic validation because value must be str.
        # All three parse attempts fail, so StructuredOutputValidationError is raised.
        client = _make_client()
        raw = '<think>hidden</think>{"value": 123}'
        with patch.object(OllamaClient, "_request", return_value=_mock_response(raw)):
            with pytest.raises(StructuredOutputValidationError):
                client.generate_structured(
                    model="test-model",
                    system_prompt="sys",
                    user_prompt="usr",
                    schema_model=_Simple,
                )

    def test_think_only_no_json_raises_structured_error(self):
        client = _make_client()
        raw = "<think>only thinking, no JSON</think>"
        with patch.object(OllamaClient, "_request", return_value=_mock_response(raw)):
            with pytest.raises(StructuredOutputValidationError):
                client.generate_structured(
                    model="qwen3.5:latest",
                    system_prompt="sys",
                    user_prompt="usr",
                    schema_model=_Simple,
                )


# ---------------------------------------------------------------------------
# generate_text — think tags stripped from plain text output
# ---------------------------------------------------------------------------


class TestGenerateText:
    def test_plain_text_returned(self):
        client = _make_client()
        with patch.object(OllamaClient, "_request", return_value=_mock_response("OK")):
            result = client.generate_text(
                model="test-model",
                system_prompt="sys",
                user_prompt="usr",
            )
        assert result == "OK"

    def test_think_prefix_stripped_from_text(self):
        client = _make_client()
        raw = "<think>\nInternal reasoning.\n</think>\nActual reply"
        with patch.object(OllamaClient, "_request", return_value=_mock_response(raw)):
            result = client.generate_text(
                model="qwen3.5:latest",
                system_prompt="sys",
                user_prompt="usr",
            )
        assert result == "Actual reply"

    def test_literal_think_string_in_text_is_preserved(self):
        client = _make_client()
        raw = "You can use <think> tags to describe reasoning."
        with patch.object(OllamaClient, "_request", return_value=_mock_response(raw)):
            result = client.generate_text(
                model="qwen3.5:latest",
                system_prompt="sys",
                user_prompt="usr",
            )
        assert result == raw


# ---------------------------------------------------------------------------
# _extract_json_object — code-fence and surrounding-prose extraction
# ---------------------------------------------------------------------------


class TestExtractJsonObject:
    def test_bare_json_returned_as_is(self):
        text = '{"value": "ok"}'
        assert OllamaClient._extract_json_object(text) == '{"value": "ok"}'

    def test_json_code_fence_extracted(self):
        text = 'Here is the output:\n```json\n{"value": "hi"}\n```'
        assert OllamaClient._extract_json_object(text) == '{"value": "hi"}'

    def test_plain_code_fence_extracted(self):
        text = 'Result:\n```\n{"value": "ok"}\n```'
        assert OllamaClient._extract_json_object(text) == '{"value": "ok"}'

    def test_json_surrounded_by_prose_extracted(self):
        text = 'Here is my response: {"value": "test"} Hope that helps!'
        assert OllamaClient._extract_json_object(text) == '{"value": "test"}'

    def test_no_json_returns_none(self):
        assert OllamaClient._extract_json_object("no json here") is None

    def test_empty_string_returns_none(self):
        assert OllamaClient._extract_json_object("") is None


# ---------------------------------------------------------------------------
# generate_structured — code-fence fallback (qwen3.5-style responses)
# ---------------------------------------------------------------------------


class TestGenerateStructuredCodeFenceFallback:
    def test_json_code_fence_parsed(self):
        client = _make_client()
        raw = 'Sure! Here is the output:\n```json\n{"value": "fenced"}\n```'
        with patch.object(OllamaClient, "_request", return_value=_mock_response(raw)):
            result = client.generate_structured(
                model="qwen3.5:latest",
                system_prompt="sys",
                user_prompt="usr",
                schema_model=_Simple,
            )
        assert result.value == "fenced"

    def test_json_with_surrounding_prose_parsed(self):
        client = _make_client()
        raw = 'My response is {"value": "prose"} as requested.'
        with patch.object(OllamaClient, "_request", return_value=_mock_response(raw)):
            result = client.generate_structured(
                model="qwen3.5:latest",
                system_prompt="sys",
                user_prompt="usr",
                schema_model=_Simple,
            )
        assert result.value == "prose"

    def test_think_plus_code_fence_parsed(self):
        client = _make_client()
        raw = '<think>I will return JSON.</think>\n```json\n{"value": "combo"}\n```'
        with patch.object(OllamaClient, "_request", return_value=_mock_response(raw)):
            result = client.generate_structured(
                model="qwen3.5:latest",
                system_prompt="sys",
                user_prompt="usr",
                schema_model=_Simple,
            )
        assert result.value == "combo"

    def test_completely_invalid_raises(self):
        client = _make_client()
        raw = "I cannot produce JSON sorry."
        with patch.object(OllamaClient, "_request", return_value=_mock_response(raw)):
            with pytest.raises(StructuredOutputValidationError) as exc_info:
                client.generate_structured(
                    model="qwen3.5:latest",
                    system_prompt="sys",
                    user_prompt="usr",
                    schema_model=_Simple,
                )
        # Full raw text should appear in the error message
        assert "I cannot produce JSON sorry." in str(exc_info.value)


# ---------------------------------------------------------------------------
# payload settings for qwen-style thinking control
# ---------------------------------------------------------------------------


class TestThinkingControlPayload:
    def test_generate_structured_sets_think_false(self):
        client = _make_client()

        def fake_request(path: str, payload: dict):
            assert path == "/api/generate"
            assert payload["think"] is False
            assert payload["options"]["thinking"] is False
            return _mock_response('{"value": "ok"}')

        with patch.object(OllamaClient, "_request", side_effect=fake_request):
            result = client.generate_structured(
                model="qwen3.5:latest",
                system_prompt="sys",
                user_prompt="usr",
                schema_model=_Simple,
            )
        assert result.value == "ok"

    def test_generate_text_sets_think_false(self):
        client = _make_client()

        def fake_request(path: str, payload: dict):
            assert path == "/api/generate"
            assert payload["think"] is False
            assert payload["options"]["thinking"] is False
            return _mock_response("OK")

        with patch.object(OllamaClient, "_request", side_effect=fake_request):
            result = client.generate_text(
                model="qwen3.5:latest",
                system_prompt="sys",
                user_prompt="usr",
            )
        assert result == "OK"

    def test_empty_response_has_hintful_error(self):
        client = _make_client()
        with patch.object(OllamaClient, "_request", return_value={"response": ""}):
            with pytest.raises(StructuredOutputValidationError) as exc_info:
                client.generate_structured(
                    model="qwen3.5:latest",
                    system_prompt="sys",
                    user_prompt="usr",
                    schema_model=_Simple,
                )
        assert "thinking output" in str(exc_info.value)
