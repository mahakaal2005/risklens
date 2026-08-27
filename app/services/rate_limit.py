"""Minimal in-memory rate limiter for local-demo use.

Single-process, in-memory sliding window -- resets on restart, not shared
across workers/replicas. Adequate for this local single-operator prototype
(see SECURITY.md); not a production rate-limiting solution (no Redis/shared
store, no distributed enforcement). Added specifically to cover the
"authentication endpoints must have rate limiting" check from the project's
standing 5-security-checks requirement.
"""

from __future__ import annotations

import time
from collections import defaultdict

_attempts: dict[str, list[float]] = defaultdict(list)


class RateLimitExceededError(Exception):
    pass


def check_rate_limit(key: str, max_attempts: int, window_seconds: float) -> None:
    """Raises RateLimitExceededError if `key` (e.g. "login:<client-ip>") has
    made max_attempts or more calls within the trailing window_seconds.
    Every call (successful or not) counts -- callers decide whether to call
    this before or after checking credentials."""
    now = time.monotonic()
    window_start = now - window_seconds
    recent = [t for t in _attempts[key] if t >= window_start]
    if len(recent) >= max_attempts:
        recent.append(now)
        _attempts[key] = recent
        raise RateLimitExceededError(f"Too many attempts for {key!r}. Try again later.")
    recent.append(now)
    _attempts[key] = recent


def reset_all() -> None:
    """Test-only helper -- clears all tracked attempts."""
    _attempts.clear()
