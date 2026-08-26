"""Simulated merchant evidence submission route.

Delegates entirely to app/services/evidence_service.py -- no repository
mutation happens directly in this route. Evidence references are validated
strings only; no real file upload or URL retrieval exists here or
anywhere in this codebase.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, raise_api_error
from app.schemas.api_responses import AuditEventResponse, ErrorCode, EvidenceSubmissionRequestBody, EvidenceSubmissionResponse
from app.services.audit_service import _assert_payload_is_safe
from app.services.case_service import CaseNotFoundError, InvalidTransitionError
from app.services.evidence_service import submit_evidence

router = APIRouter(tags=["evidence"])


@router.post("/cases/{case_id}/evidence", response_model=EvidenceSubmissionResponse)
def post_evidence_route(case_id: str, body: EvidenceSubmissionRequestBody, db: Session = Depends(get_db)) -> EvidenceSubmissionResponse:
    if not body.merchant_actor_id or not body.merchant_actor_id.strip():
        raise_api_error(422, ErrorCode.VALIDATION_ERROR, "merchant_actor_id must not be empty.")

    try:
        evidence, audit_event = submit_evidence(
            db, case_id, body.merchant_explanation_text, body.evidence_references, actor_id=body.merchant_actor_id,
        )
    except CaseNotFoundError:
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No review case exists for the provided case ID.")
    except InvalidTransitionError as exc:
        raise_api_error(409, ErrorCode.INVALID_CASE_TRANSITION, str(exc))
    except ValueError as exc:
        raise_api_error(422, ErrorCode.INVALID_EVIDENCE_REFERENCE, str(exc))

    _assert_payload_is_safe(audit_event["event_payload_json"])  # defense-in-depth re-check at read time
    new_audit_event = AuditEventResponse(
        event_sequence_number=audit_event["event_sequence_number"],
        event_timestamp=audit_event["event_timestamp"],
        actor_type=audit_event["actor_type"],
        actor_id=audit_event["actor_id"],
        event_type=audit_event["event_type"],
        event_payload=audit_event["event_payload_json"],
    )

    return EvidenceSubmissionResponse(
        evidence_id=evidence.evidence_id,
        case_id=evidence.case_id,
        case_status=evidence.case.case_status,
        submitted_at=evidence.submitted_at,
        evidence_references=evidence.evidence_references_json,
        new_audit_event=new_audit_event,
    )
