from __future__ import annotations

from pathlib import Path

from app.utils.markdown_logger import MarkdownLogger


def test_write_result_snapshot_creates_markdown(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    logger = MarkdownLogger(output_dir=log_dir, result_dir=out_dir)

    path = logger.write_result_snapshot(
        session_id="sid123",
        topic="topic",
        final_summary="summary text",
        input_sources=["a.md", "b.md"],
        validation_highlights=["issue-1", "issue-2"],
    )

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Debate Result sid123" in content
    assert "topic" in content
    assert "summary text" in content
    assert "a.md" in content
    assert "b.md" in content
    assert "issue-1" in content
