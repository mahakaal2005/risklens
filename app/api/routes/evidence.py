"""Simulated merchant evidence submission, plus real evidence-attachment
upload/download (Phase 2 -- see docs/PHASE_2_EVIDENCE_ATTACHMENTS_DESIGN.md).

Delegates entirely to app/services/evidence_service.py and
app/services/evidence_attachment_service.py -- no repository mutation
happens directly in this route. Evidence references remain validated
strings only; attachments are real files, stored locally, validated by
extension allowlist + size cap + magic-byte content check, never served
or stored under a client-supplied filename/path.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db, raise_api_error, require_role
from app.db.models import User
from app.db.repositories import get_case, get_evidence_submission
from app.schemas.api_responses import (
    AttachmentUploadResponse,
    AuditEventResponse,
    ErrorCode,
    EvidenceAttachmentSummary,
    EvidenceSubmissionRequestBody,
    EvidenceSubmissionResponse,
)
from app.services.audit_service import _assert_payload_is_safe, record_event
from app.services.case_service import CaseNotFoundError, InvalidTransitionError
from app.services.evidence_attachment_service import InvalidAttachmentError, read_attachment_bytes, save_attachment
from app.services.evidence_service import submit_evidence

router = APIRouter(tags=["evidence"])

MAX_UPLOAD_READ_BYTES = 5 * 1024 * 1024 + 1  # one byte over the service's own cap, so an oversized
# upload is rejected without ever buffering an unbounded amount of attacker-controlled data.


def _get_owned_evidence(db: Session, case_id: str, evidence_id: str, user: User):
    """Shared ownership check for both attachment routes: the case must
    exist, belong to the caller's merchant_id, and the evidence_id must
    belong to that same case. Any mismatch is a 404 -- never a 403 -- so no
    response reveals whether a case/evidence_id it doesn't own even exists."""
    case = get_case(db, case_id)
    if case is None or case.merchant_id != user.merchant_id:
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No review case exists for the provided case ID.")

    evidence = get_evidence_submission(db, evidence_id)
    if evidence is None or evidence.case_id != case_id:
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No evidence submission exists for the provided ID.")
    return evidence


@router.post("/cases/{case_id}/evidence", response_model=EvidenceSubmissionResponse)
def post_evidence_route(
    case_id: str,
    body: EvidenceSubmissionRequestBody,
    db: Session = Depends(get_db),
    user: User = Depends(require_role("merchant")),
) -> EvidenceSubmissionResponse:
    case = get_case(db, case_id)
    if case is None:
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No review case exists for the provided case ID.")
    if case.merchant_id != user.merchant_id:
        # 404, not 403: a merchant gets no signal that a case belonging to
        # a different merchant even exists.
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No review case exists for the provided case ID.")

    try:
        evidence, audit_event = submit_evidence(
            db, case_id, body.merchant_explanation_text, body.evidence_references, actor_id=user.actor_id,
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


@router.post("/cases/{case_id}/evidence/{evidence_id}/attachments", response_model=AttachmentUploadResponse)
async def post_evidence_attachment_route(
    case_id: str,
    evidence_id: str,
    file: UploadFile,
    db: Session = Depends(get_db),
    user: User = Depends(require_role("merchant")),
) -> AttachmentUploadResponse:
    evidence = _get_owned_evidence(db, case_id, evidence_id, user)

    content = await file.read(MAX_UPLOAD_READ_BYTES)
    if len(content) >= MAX_UPLOAD_READ_BYTES:
        raise_api_error(422, ErrorCode.INVALID_EVIDENCE_REFERENCE, "File exceeds the maximum allowed size of 5 MB.")

    try:
        attachment = save_attachment(db, evidence, file.filename or "", content, file.content_type or "")
    except InvalidAttachmentError as exc:
        raise_api_error(422, ErrorCode.INVALID_EVIDENCE_REFERENCE, str(exc))

    audit_event = record_event(
        db,
        case_id,
        "merchant_demo",
        user.actor_id,
        "EVIDENCE_ATTACHMENT_UPLOADED",
        {
            "evidence_id": evidence_id,
            "attachment_id": attachment.attachment_id,
            "original_filename": attachment.original_filename,
            "content_type": attachment.content_type,
            "size_bytes": attachment.size_bytes,
        },
    )

    return AttachmentUploadResponse(
        attachment=EvidenceAttachmentSummary(
            attachment_id=attachment.attachment_id,
            original_filename=attachment.original_filename,
            content_type=attachment.content_type,
            size_bytes=attachment.size_bytes,
            uploaded_at=attachment.uploaded_at,
        ),
        new_audit_event=AuditEventResponse(
            event_sequence_number=audit_event["event_sequence_number"],
            event_timestamp=audit_event["event_timestamp"],
            actor_type=audit_event["actor_type"],
            actor_id=audit_event["actor_id"],
            event_type=audit_event["event_type"],
            event_payload=audit_event["event_payload_json"],
        ),
    )


@router.get("/cases/{case_id}/evidence/{evidence_id}/attachments/{attachment_id}")
def get_evidence_attachment_route(
    case_id: str,
    evidence_id: str,
    attachment_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    """Download route. Any authenticated role may read (reviewer/risk_manager
    for oversight, merchant for their own case) -- merchant-role callers are
    scoped to their own merchant_id exactly like every other case read."""
    case = get_case(db, case_id)
    if case is None:
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No review case exists for the provided case ID.")
    if user.role == "merchant" and case.merchant_id != user.merchant_id:
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No review case exists for the provided case ID.")

    evidence = get_evidence_submission(db, evidence_id)
    if evidence is None or evidence.case_id != case_id:
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No evidence submission exists for the provided ID.")

    attachment = next((a for a in evidence.attachments if a.attachment_id == attachment_id), None)
    if attachment is None:
        raise_api_error(404, ErrorCode.CASE_NOT_FOUND, "No attachment exists for the provided ID.")

    content = read_attachment_bytes(attachment)
    # original_filename is client-supplied and only ever meant for display --
    # strip CR/LF (header/response-splitting) and quotes (breaking out of the
    # quoted-string header value) before it ever reaches a response header.
    safe_display_name = attachment.original_filename.replace("\r", "").replace("\n", "").replace('"', "")
    return Response(
        content=content,
        media_type=attachment.content_type,
        headers={"Content-Disposition": f'attachment; filename="{safe_display_name}"'},
    )
