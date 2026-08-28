"""Unit tests for app/services/rate_limit.py -- the in-memory sliding-window
limiter added to satisfy the project's standing security-checklist
requirement that authentication endpoints be rate limited."""

from __future__ import annotations

import time

import pytest

from app.services.rate_limit import RateLimitExceededError, check_rate_limit, reset_all


@pytest.fixture(autouse=True)
def _reset():
    reset_all()
    yield
    reset_all()


def test_allows_calls_under_the_limit():
    for _ in range(4):
        check_rate_limit("k1", max_attempts=5, window_seconds=60)  # no raise


def test_blocks_the_call_at_the_limit():
    for _ in range(5):
        check_rate_limit("k2", max_attempts=5, window_seconds=60)
    with pytest.raises(RateLimitExceededError):
        check_rate_limit("k2", max_attempts=5, window_seconds=60)


def test_different_keys_are_tracked_independently():
    for _ in range(5):
        check_rate_limit("k3a", max_attempts=5, window_seconds=60)
    check_rate_limit("k3b", max_attempts=5, window_seconds=60)  # different key, no raise


def test_old_attempts_outside_the_window_do_not_count():
    for _ in range(5):
        check_rate_limit("k4", max_attempts=5, window_seconds=0.05)
    time.sleep(0.1)
    check_rate_limit("k4", max_attempts=5, window_seconds=0.05)  # window has passed, no raise
