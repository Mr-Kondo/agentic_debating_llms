"""Input helpers for loading debate topics from markdown files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


class InputSourceError(RuntimeError):
    """Raised when no usable input markdown files are found."""


@dataclass(slots=True)
class InputPayload:
    """Resolved debate input from one or more markdown sources."""

    topic: str
    details: str
    sources: list[str]


def _first_meaningful_line(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^#+\s*", "", line)
        if line:
            return line
    return ""


def load_input_payload(input_dir: Path, pattern: str = "*.md") -> InputPayload:
    """Load all markdown files and produce a topic plus details text.

    Files are concatenated in lexical filename order to keep output deterministic.
    """
    base = input_dir.expanduser().resolve()
    files = sorted([p for p in base.glob(pattern) if p.is_file()])
    if not files:
        raise InputSourceError(
            f"No markdown files found in {base}. Provide --topic or place .md files under the input directory."
        )

    chunks: list[str] = []
    sources: list[str] = []
    first_source_topic = ""
    for path in files:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        if not first_source_topic:
            first_source_topic = _first_meaningful_line(content)
        chunks.append(f"# Source: {path.name}\n\n{content}")
        sources.append(path.name)

    if not chunks:
        raise InputSourceError(f"Markdown files exist in {base}, but all were empty.")

    details = "\n\n---\n\n".join(chunks)
    topic = first_source_topic or _first_meaningful_line(details)
    if not topic:
        topic = "Discussion topic from input markdown"

    if len(topic) > 120:
        topic = topic[:117] + "..."

    return InputPayload(topic=topic, details=details, sources=sources)
