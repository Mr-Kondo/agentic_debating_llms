"""Entry point: ``uv run python -m app.licenses``

Generates ``docs/third_party_licenses.md`` from ``uv.lock`` and installed
distribution metadata in the project virtual environment.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_site_packages(repo_root: Path) -> Path:
    venv_lib = repo_root / ".venv" / "lib"
    if not venv_lib.is_dir():
        print("ERROR: .venv/lib not found. Run `uv sync` first.", file=sys.stderr)
        sys.exit(1)

    candidates = sorted(venv_lib.glob("python3.*"))
    if not candidates:
        print("ERROR: No python3.x directory found under .venv/lib", file=sys.stderr)
        sys.exit(1)

    return candidates[-1] / "site-packages"


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    lock_path = repo_root / "uv.lock"

    if not lock_path.exists():
        print(f"ERROR: uv.lock not found at {lock_path}", file=sys.stderr)
        sys.exit(1)

    site_packages = _find_site_packages(repo_root)

    # Import here so the module is usable without installation
    from app.licenses.collector import collect
    from app.licenses.renderer import write_markdown

    packages = collect(lock_path, site_packages)
    out_path = repo_root / "docs" / "third_party_licenses.md"
    write_markdown(packages, out_path)

    print(f"Written {out_path} ({len(packages)} runtime packages)")


if __name__ == "__main__":
    main()
