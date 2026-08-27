"""Case list, detail, audit-timeline, and reviewer-action routes.

No route here mutates the database directly -- every state-changing
endpoint calls the existing app/services/case_service.py so state-machine
enforcement and audit-event creation stay in one place. Local
synthetic-data demonstration only.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db, raise_api_error, require_role
from app.db.models import EvidenceSubmission, ReviewCase, User
from app.db.repositories import get_audit_events_for_case, get_case, list_cases
from app.schemas.api_responses import (
    AuditEventResponse,
    AuditTimelineResponse,
    CaseDetailResponse,
    CaseListResponse,
    CaseSummary,
    ErrorCode,
    EvidenceAttachmentSummary,
    EvidenceSubmissionSummary,
    ReviewActionRequest,
    ReviewActionResponse,
)
from app.services.audit_service import _assert_payload_is_safe
from app.services.case_service import CaseNotFoundError, InvalidTransitionError, apply_reviewer_action, start_review

router = APIRouter(tags=["cases"])


def _to_case_summary(case: ReviewCase) -> CaseSummary:
    return CaseSummary(
        case_id=case.case_id,
        merchant_id=case.merchant_id,
        week_start=case.week_start,
        case_status=case.case_status,
        risk_signal_intensity=case.risk_signal_intensity,
        recommendation=case.recommendation,
        created_at=case.created_at,
        updated_at=case.updated_at,
        final_outcome=case.final_outcome,
    )


def _to_evidence_summary(evidence: EvidenceSubmission) -> EvidenceSubmissionSummary:
    return EvidenceSubmissionSummary(
        evidence_id=evidence.evidence_id,
        submitted_at=evidence.submitted_at,
        status=evidence.status,
        evidence_references=evidence.evidence_references_json,
        attachments=[
            EvidenceAttachmentSummary(
                attachment_id=a.attachment_id,
                original_filename=a.original_filename,
                content_type=a.content_type,
                size_bytes=a.size_bytes,
                uploaded_at=a.uploaded_at,
            )
            for a in evidence.attachments
        ],
    )


def _to_case_detail(case: ReviewCase) -> CaseDetailResponse:
    return CaseDetailResponse(
        case_id=case.case_id,
        merchant_id=case.merchant_id,
        week_start=case.week_start,
        case_status=case.case_status,
        risk_signal_intensity=case.risk_signal_intensity,
        recommendation=case.recommendation,
        policy_explanation=case.policy_explanation,
        model_probability=case.model_probability,
        rules_only_score=case.rules_only_score,
        analyst_summary=case.analyst_summary,
        merchant_safe_explanation=case.merchant_safe_explanation,
        triggered_rules=case.triggered_rules_json,
        evidence_checklist=case.evidence_checklist_json,
        model_version=case.model_version,
        rules_version=case.rules_version,
        created_at=case.created_at,
        updated_at=case.updated_at,
        resolved_at=case.resolved_at,
        final_outcome=case.final_outcome,
        reviewer_note=case.reviewer_note,
        evidence_submissions=[_to_evidence_summary(e) for e in case.evidence_submissions],
    )


def _to_audit_event_response(event) -> AuditEventResponse:
    payload = event.event_payload_json
    _assert_payload_is_safe(payload)  # defense-in-depth re-check at read time
    return AuditEventResponse(
        event_sequence_number=event.event_sequence_number,
        event_timestamp=event.event_timestamp,
        actor_type=event.actor_type,
        actor_id=event.actor_id,
        event_type=event.event_type,
        event_payload=payload,
    )


def _enforce_merchant_scope(user: User, case: ReviewCase | None) -> None:
    """A merchant-role user may only read cases belonging to their own
    merchant_id -- see docs/PHASE_2_AUTH_DESIGN.md Section 5. Reviewer and
    risk_manager roles are unrestricted readers. Returns a 404 (not 403)
    for a merchant reading someone else's case, so the response gives no
    signal about whether that case exists at all."""
    if user.role == "merchant" and case is not None and case.merchant_id != user.merchant_id:
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No review case exists for the provided case ID.")


@router.get("/cases", response_model=CaseListResponse)
def list_cases_route(
    status: str | None = Query(default=None),
    recommendation: str | None = Query(default=None),
    intensity: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> CaseListResponse:
    merchant_scope = user.merchant_id if user.role == "merchant" else None
    cases, total = list_cases(
        db, status=status, recommendation=recommendation, intensity=intensity, limit=limit, offset=offset,
        merchant_id=merchant_scope,
    )
    return CaseListResponse(items=[_to_case_summary(c) for c in cases], limit=limit, offset=offset, total=total)


@router.get("/cases/{case_id}", response_model=CaseDetailResponse)
def get_case_detail_route(case_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> CaseDetailResponse:
    case = get_case(db, case_id)
    if case is None:
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No review case exists for the provided case ID.")
    _enforce_merchant_scope(user, case)
    return _to_case_detail(case)


@router.get("/cases/{case_id}/audit-events", response_model=AuditTimelineResponse)
def get_case_audit_events_route(
    case_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> AuditTimelineResponse:
    case = get_case(db, case_id)
    if case is None:
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No review case exists for the provided case ID.")
    _enforce_merchant_scope(user, case)
    events = get_audit_events_for_case(db, case_id)
    return AuditTimelineResponse(case_id=case_id, events=[_to_audit_event_response(e) for e in events])


@router.post("/cases/{case_id}/review-actions", response_model=ReviewActionResponse)
def post_review_action_route(
    case_id: str,
    body: ReviewActionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_role("reviewer")),
) -> ReviewActionResponse:
    if not body.reviewer_note or not body.reviewer_note.strip():
        raise_api_error(422, ErrorCode.VALIDATION_ERROR, "reviewer_note must not be empty.")

    events_before = len(get_audit_events_for_case(db, case_id)) if get_case(db, case_id) else 0

    try:
        if body.action.value == "START_REVIEW":
            case = start_review(db, case_id, reviewer_actor=user.actor_id)
        else:
            case = apply_reviewer_action(db, case_id, body.action.value, body.reviewer_note, reviewer_actor=user.actor_id)
    except CaseNotFoundError:
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No review case exists for the provided case ID.")
    except InvalidTransitionError as exc:
        raise_api_error(409, ErrorCode.INVALID_CASE_TRANSITION, str(exc))
    except ValueError as exc:
        raise_api_error(422, ErrorCode.VALIDATION_ERROR, str(exc))

    all_events = get_audit_events_for_case(db, case_id)
    new_events = all_events[events_before:]
    return ReviewActionResponse(case=_to_case_detail(case), new_audit_events=[_to_audit_event_response(e) for e in new_events])
