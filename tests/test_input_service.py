from __future__ import annotations

from pathlib import Path

import pytest

from app.services.input_service import InputSourceError, load_input_payload


def test_load_input_payload_concatenates_files_in_lexical_order(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "b.md").write_text("## B title\nbody-b", encoding="utf-8")
    (input_dir / "a.md").write_text("# A title\nbody-a", encoding="utf-8")

    payload = load_input_payload(input_dir)

    assert payload.sources == ["a.md", "b.md"]
    assert payload.topic == "A title"
    assert "body-a" in payload.details
    assert "body-b" in payload.details


def test_load_input_payload_raises_when_no_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    with pytest.raises(InputSourceError):
        load_input_payload(input_dir)


def test_load_input_payload_raises_when_all_files_empty(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "a.md").write_text("\n\n", encoding="utf-8")

    with pytest.raises(InputSourceError):
        load_input_payload(input_dir)
