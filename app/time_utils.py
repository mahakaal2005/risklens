"""Shared UTC datetime helpers.

Previously duplicated identically across app/db/models.py,
app/services/auth_service.py, and app/services/sla_service.py.
"""

from __future__ import annotations

import datetime as dt


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def as_aware_utc(value: dt.datetime) -> dt.datetime:
    """SQLite has no native timezone-aware column type -- a DateTime(timezone=True)
    value written as UTC can still come back naive on read, depending on
    driver/dialect behavior. Treat any naive value as UTC rather than
    letting it silently compare wrong (or raise) against an aware value."""
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.timezone.utc)
