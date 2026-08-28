"""Case-workflow service: creates review cases from Milestone 4 case packets
and enforces the documented finite state machine for reviewer actions.

This remains a local synthetic-data demonstration: no automated
enforcement action (freeze/ban/terminate/reject/hold) exists anywhere in
this module, and label_high_loss_next_30d / latent_state_for_demo_only are
defensively asserted absent from every packet before persistence.
"""

from __future__ import annotations

import datetime as dt
import json

from sqlalchemy.orm import Session

from app.db.models import ReviewCase
from app.db.repositories import add_case, get_case, get_case_by_merchant_week
from app.services.audit_service import record_event
from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN

APPROVE = "APPROVE"

# state -> {reviewer_action: next_state}
VALID_TRANSITIONS: dict[str, dict[str, str]] = {
    "OPEN": {
        "REQUEST_EVIDENCE": "EVIDENCE_REQUESTED",
        "CLEAR_CASE": "RESOLVED",
        "MARK_FALSE_POSITIVE": "RESOLVED",
        "MARK_OPERATIONAL_ISSUE": "RESOLVED",
        "MARK_INCONCLUSIVE": "RESOLVED",
        "ESCALATE_CASE": "ESCALATED",
    },
    "EVIDENCE_REQUESTED": {
        "ESCALATE_CASE": "ESCALATED",
    },
    "EVIDENCE_SUBMITTED": {},  # only start_review() is valid; see that function
    "UNDER_REVIEW": {
        "CLEAR_CASE": "RESOLVED",
        "MARK_FALSE_POSITIVE": "RESOLVED",
        "MARK_OPERATIONAL_ISSUE": "RESOLVED",
        "MARK_INCONCLUSIVE": "RESOLVED",
        "ESCALATE_CASE": "ESCALATED",
    },
    "ESCALATED": {},  # no further automated action; see add_escalation_note()
    "RESOLVED": {},  # immutable
}

ACTION_TO_FINAL_OUTCOME = {
    "CLEAR_CASE": "CONFIRMED_RISK",
    "MARK_FALSE_POSITIVE": "FALSE_POSITIVE",
    "MARK_OPERATIONAL_ISSUE": "OPERATIONAL_ISSUE",
    "MARK_INCONCLUSIVE": "INCONCLUSIVE",
}

# EVIDENCE_REQUEST_RECOMMENDED is reused for both the system's initial
# suggestion (actor_type=system, at case creation) and the reviewer's
# explicit REQUEST_EVIDENCE action (actor_type=analyst_demo) -- documented
# in docs/AUDIT_EVENT_SCHEMA.md.
ACTION_TO_AUDIT_EVENT_TYPE = {
    "CLEAR_CASE": "CASE_CLEARED",
    "MARK_FALSE_POSITIVE": "CASE_MARKED_FALSE_POSITIVE",
    "MARK_OPERATIONAL_ISSUE": "CASE_MARKED_OPERATIONAL_ISSUE",
    "MARK_INCONCLUSIVE": "CASE_MARKED_INCONCLUSIVE",
    "ESCALATE_CASE": "CASE_ESCALATED",
    "REQUEST_EVIDENCE": "EVIDENCE_REQUEST_RECOMMENDED",
}


class InvalidTransitionError(Exception):
    pass


class CaseNotFoundError(Exception):
    pass


def _assert_packet_has_no_leakage(packet: dict) -> None:
    serialized = json.dumps(packet)
    if LABEL_COLUMN in serialized or LATENT_STATE_COLUMN in serialized:
        raise ValueError("Case packet must never contain the target label or latent state before persistence.")


def create_case_from_packet(session: Session, packet: dict) -> tuple[ReviewCase | None, list[dict]]:
    """Creates a ReviewCase only when the recommendation is not APPROVE.
    Returns (None, []) for an APPROVE packet -- no case, no audit events."""
    _assert_packet_has_no_leakage(packet)

    assessment = packet["assessment"]
    recommendation = assessment["recommendation"]
    if recommendation == APPROVE:
        return None, []

    identification = packet["identification"]
    existing = get_case_by_merchant_week(session, identification["merchant_id"], identification["week_start"])
    if existing is not None:
        return existing, []

    case = ReviewCase(
        case_id=identification["case_preview_id"],
        merchant_id=identification["merchant_id"],
        week_start=identification["week_start"],
        case_status="OPEN",
        risk_signal_intensity=assessment["risk_signal_intensity"],
        model_probability=assessment["model_probability"],
        selected_threshold=assessment["selected_threshold"],
        rules_only_score=assessment["rules_only_score"],
        recommendation=recommendation,
        policy_explanation=assessment["policy_explanation"],
        analyst_summary=packet["analyst_explanation"]["summary"],
        merchant_safe_explanation=packet["merchant_safe_explanation"],
        triggered_rules_json=assessment["triggered_rules"],
        triggered_rule_explanations_json=packet["analyst_explanation"].get("triggered_rule_explanations", []),
        evidence_checklist_json=packet["evidence_checklist"],
        model_version=assessment["model_version"],
        rules_version=assessment["rules_version"],
        synthetic_data_notice=identification["synthetic_data_notice"],
    )
    add_case(session, case)

    audit_events = [
        record_event(session, case.case_id, "system", "system", "ASSESSMENT_GENERATED", {
            "risk_signal_intensity": assessment["risk_signal_intensity"],
            "rules_only_score": assessment["rules_only_score"],
            "model_probability": assessment["model_probability"],
        }),
        record_event(session, case.case_id, "system", "system", "EXPLANATION_GENERATED", {
            "summary": packet["analyst_explanation"]["summary"],
        }),
        record_event(session, case.case_id, "system", "system", "REVIEW_CASE_CREATED", {
            "case_id": case.case_id, "recommendation": recommendation,
        }),
        record_event(session, case.case_id, "system", "system", "REVIEW_CASE_RECOMMENDED", {
            "recommendation": recommendation, "policy_explanation": assessment["policy_explanation"],
        }),
    ]
    if recommendation == "REQUEST_EVIDENCE":
        audit_events.append(record_event(session, case.case_id, "system", "system", "EVIDENCE_REQUEST_RECOMMENDED", {
            "evidence_checklist": packet["evidence_checklist"],
        }))
    if recommendation == "MANUAL_REVIEW_REQUIRED":
        audit_events.append(record_event(session, case.case_id, "system", "system", "MANUAL_REVIEW_RECOMMENDED", {}))

    return case, audit_events


def apply_reviewer_action(session: Session, case_id: str, action: str, note: str, reviewer_actor: str = "analyst_demo") -> ReviewCase:
    if not note or not note.strip():
        raise ValueError("Reviewer note must not be empty.")

    case = get_case(session, case_id)
    if case is None:
        raise CaseNotFoundError(f"No case found for case_id={case_id}")

    if case.case_status == "RESOLVED":
        raise InvalidTransitionError("Resolved cases are immutable; no further reviewer decision can overwrite them.")

    next_status = VALID_TRANSITIONS.get(case.case_status, {}).get(action)
    if next_status is None:
        raise InvalidTransitionError(f"Action {action!r} is not valid from status {case.case_status!r}.")

    previous_status = case.case_status
    case.case_status = next_status
    case.reviewer_note = note
    case.reviewer_actor = reviewer_actor

    if action in ACTION_TO_FINAL_OUTCOME:
        case.final_outcome = ACTION_TO_FINAL_OUTCOME[action]
        case.resolved_at = _now()
        case.decision_timestamp = _now()
    elif action == "ESCALATE_CASE":
        case.decision_timestamp = _now()

    session.flush()

    record_event(session, case.case_id, "analyst_demo", reviewer_actor, ACTION_TO_AUDIT_EVENT_TYPE[action], {
        "action": action, "note": note, "previous_status": previous_status, "new_status": next_status,
    })
    return case


def start_review(session: Session, case_id: str, reviewer_actor: str = "analyst_demo") -> ReviewCase:
    case = get_case(session, case_id)
    if case is None:
        raise CaseNotFoundError(f"No case found for case_id={case_id}")
    if case.case_status != "EVIDENCE_SUBMITTED":
        raise InvalidTransitionError("Review can only begin from EVIDENCE_SUBMITTED status.")

    case.case_status = "UNDER_REVIEW"
    session.flush()
    record_event(session, case.case_id, "analyst_demo", reviewer_actor, "REVIEW_STARTED", {})
    return case


def add_escalation_note(session: Session, case_id: str, note: str, reviewer_actor: str = "analyst_demo") -> ReviewCase:
    """Escalated cases permit no further automated action -- only an
    additional note appended to the audit trail."""
    if not note or not note.strip():
        raise ValueError("Escalation note must not be empty.")
    case = get_case(session, case_id)
    if case is None:
        raise CaseNotFoundError(f"No case found for case_id={case_id}")
    if case.case_status != "ESCALATED":
        raise InvalidTransitionError("Escalation notes can only be added to an ESCALATED case.")

    record_event(session, case.case_id, "analyst_demo", reviewer_actor, "CASE_ESCALATED", {"note": note})
    return case


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)
