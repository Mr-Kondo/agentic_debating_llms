"""Time helper utilities."""

from __future__ import annotations

from datetime import datetime, timezone


def now_utc() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)
