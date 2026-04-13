"""Render collected license information to Markdown."""

from __future__ import annotations

from pathlib import Path

from app.licenses.collector import PackageInfo

_HEADER = """\
# Third-Party Licenses

This document lists all open-source libraries used at runtime by **agentic-debating-llms**
and their respective licenses.

> **Auto-generated** — do not edit by hand.
> To regenerate, run:
>
> ```bash
> uv run python -m app.licenses
> ```

## Runtime Dependencies

| Package | Version | License | Project |
|---------|---------|---------|---------|
"""

_FOOTER = """
---

*Generated from `uv.lock` and installed distribution metadata.*
"""


def render_markdown(packages: list[PackageInfo]) -> str:
    """Return a Markdown string listing *packages* in a table."""
    rows: list[str] = [_HEADER]
    for pkg in packages:
        project_cell = f"[link]({pkg.home_url})" if pkg.home_url else "—"
        rows.append(f"| {pkg.name} | {pkg.version} | `{pkg.license}` | {project_cell} |")
    rows.append(_FOOTER)
    return "\n".join(rows)


def write_markdown(packages: list[PackageInfo], out_path: Path) -> None:
    """Write *packages* as a Markdown license table to *out_path*."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(packages), encoding="utf-8")
