"""Review SLA computation (Phase 2).

No background scheduler, no stored deadline column -- the SLA status is
computed at read time from the case's existing recommendation, created_at,
and case_status fields. This matches the project's "prefer the simplest
architecture" principle and avoids a new always-running process for what is,
today, a pure display concern.

"Notification simulation" (per CLAUDE.md's Phase 2 roadmap wording) is a
simulated in-app breach indicator only -- there is no real email/SMS/webhook
integration anywhere in this codebase, and none is added here.
"""

from __future__ import annotations

import datetime as dt

from app.time_utils import as_aware_utc, utcnow

# Hours from case creation until the case is considered SLA-breached, keyed
# by the case's recommendation. Cases with no configured SLA (e.g. a
# recommendation that never results in a persisted review case) return
# sla_hours=None -- "not applicable", not a silent zero.
SLA_HOURS_BY_RECOMMENDATION: dict[str, int] = {
    "ESCALATE_TO_COMPLIANCE": 24,
    "MANUAL_REVIEW_REQUIRED": 48,
    "REQUEST_EVIDENCE": 72,
}

# Once a case reaches one of these statuses, its SLA clock has stopped --
# it was handled (RESOLVED) or handed off for compliance follow-up
# (ESCALATED), so "breached" no longer applies.
TERMINAL_STATUSES = {"RESOLVED", "ESCALATED"}


def compute_sla(recommendation: str, created_at: dt.datetime, case_status: str) -> dict:
    """Returns a dict with sla_hours, sla_deadline, hours_until_deadline,
    and sla_breached -- all None/False when no SLA applies or the case's
    clock has already stopped."""

    sla_hours = SLA_HOURS_BY_RECOMMENDATION.get(recommendation)
    if sla_hours is None:
        return {"sla_hours": None, "sla_deadline": None, "hours_until_deadline": None, "sla_breached": False}

    created_at = as_aware_utc(created_at)
    deadline = created_at + dt.timedelta(hours=sla_hours)

    if case_status in TERMINAL_STATUSES:
        return {"sla_hours": sla_hours, "sla_deadline": deadline, "hours_until_deadline": None, "sla_breached": False}

    remaining = (deadline - utcnow()).total_seconds() / 3600.0
    return {
        "sla_hours": sla_hours,
        "sla_deadline": deadline,
        "hours_until_deadline": round(remaining, 2),
        "sla_breached": remaining < 0,
    }
