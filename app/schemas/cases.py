"""Pydantic schemas for the review-case workflow: the finite state machine,
allowed reviewer actions/outcomes, and the case read model.

See docs/MILESTONE_5_CASE_WORKFLOW.md for the full state-transition table.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class CaseStatus(str, Enum):
    OPEN = "OPEN"
    EVIDENCE_REQUESTED = "EVIDENCE_REQUESTED"
    EVIDENCE_SUBMITTED = "EVIDENCE_SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"


class FinalOutcome(str, Enum):
    CONFIRMED_RISK = "CONFIRMED_RISK"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    OPERATIONAL_ISSUE = "OPERATIONAL_ISSUE"
    INCONCLUSIVE = "INCONCLUSIVE"


class ReviewerAction(str, Enum):
    CLEAR_CASE = "CLEAR_CASE"
    MARK_FALSE_POSITIVE = "MARK_FALSE_POSITIVE"
    REQUEST_EVIDENCE = "REQUEST_EVIDENCE"
    MARK_OPERATIONAL_ISSUE = "MARK_OPERATIONAL_ISSUE"
    ESCALATE_CASE = "ESCALATE_CASE"
    MARK_INCONCLUSIVE = "MARK_INCONCLUSIVE"


# Prohibited outcomes/actions that must never exist in this codebase, kept
# here only as a documented negative list for tests to assert against.
PROHIBITED_OUTCOMES = {"BANNED", "FROZEN", "TERMINATED", "PAYMENT_REJECTED", "SETTLEMENT_HELD"}


class ReviewerActionRequest(BaseModel):
    action: ReviewerAction
    note: str
    reviewer_actor: str = "analyst_demo"

    @field_validator("note")
    @classmethod
    def note_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Reviewer note must not be empty.")
        return v


class ReviewCaseRead(BaseModel):
    case_id: str
    merchant_id: str
    week_start: str
    case_status: CaseStatus
    risk_signal_intensity: str
    model_probability: float | None
    selected_threshold: float
    rules_only_score: int
    recommendation: str
    policy_explanation: str
    analyst_summary: str
    merchant_safe_explanation: dict
    triggered_rules_json: list[str]
    evidence_checklist_json: list[str]
    model_version: str | None
    rules_version: str
    synthetic_data_notice: str
    created_at: dt.datetime
    updated_at: dt.datetime
    resolved_at: dt.datetime | None
    final_outcome: FinalOutcome | None
    reviewer_note: str | None
    reviewer_actor: str | None
    decision_timestamp: dt.datetime | None

    model_config = {"from_attributes": True}
