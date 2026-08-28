"""Unit tests for app/services/sla_service.py -- computed (not stored)
review SLA status. See docs/PHASE_2_REVIEW_SLA_DESIGN.md."""

from __future__ import annotations

import datetime as dt

from app.services.sla_service import compute_sla


def _hours_ago(hours: float) -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)


def test_recommendation_with_no_configured_sla_returns_not_applicable():
    result = compute_sla("ALLOW_WITH_MONITORING", _hours_ago(1), "OPEN")
    assert result["sla_hours"] is None
    assert result["sla_deadline"] is None
    assert result["hours_until_deadline"] is None
    assert result["sla_breached"] is False


def test_case_within_sla_window_is_not_breached():
    result = compute_sla("REQUEST_EVIDENCE", _hours_ago(1), "OPEN")
    assert result["sla_hours"] == 72
    assert result["sla_breached"] is False
    assert result["hours_until_deadline"] > 0


def test_case_past_deadline_is_breached():
    result = compute_sla("REQUEST_EVIDENCE", _hours_ago(100), "OPEN")
    assert result["sla_breached"] is True
    assert result["hours_until_deadline"] < 0


def test_escalate_to_compliance_has_the_shortest_window():
    result = compute_sla("ESCALATE_TO_COMPLIANCE", _hours_ago(1), "OPEN")
    assert result["sla_hours"] == 24


def test_manual_review_required_window():
    result = compute_sla("MANUAL_REVIEW_REQUIRED", _hours_ago(1), "OPEN")
    assert result["sla_hours"] == 48


def test_resolved_case_stops_the_clock_even_if_created_long_ago():
    result = compute_sla("MANUAL_REVIEW_REQUIRED", _hours_ago(1000), "RESOLVED")
    assert result["sla_hours"] == 48  # still reported, for display
    assert result["sla_breached"] is False
    assert result["hours_until_deadline"] is None


def test_escalated_case_stops_the_clock():
    result = compute_sla("ESCALATE_TO_COMPLIANCE", _hours_ago(1000), "ESCALATED")
    assert result["sla_breached"] is False
    assert result["hours_until_deadline"] is None


def test_naive_datetime_is_treated_as_utc():
    naive_created_at = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).replace(tzinfo=None)
    result = compute_sla("REQUEST_EVIDENCE", naive_created_at, "OPEN")
    assert result["sla_breached"] is False  # would raise if naive/aware comparison were broken
