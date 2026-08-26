"""Simulated merchant evidence submission.

Evidence references are validated safe demo strings only -- no real file
upload, external URL retrieval, or document storage exists here. Evidence
can only be submitted while a case is in EVIDENCE_REQUESTED status.
"""

from __future__ import annotations

import uuid

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.db.models import EvidenceSubmission
from app.db.repositories import add_evidence, get_case
from app.schemas.evidence import EvidenceSubmissionRequest
from app.services.audit_service import record_event
from app.services.case_service import CaseNotFoundError, InvalidTransitionError


def submit_evidence(
    session: Session,
    case_id: str,
    merchant_explanation_text: str,
    evidence_references: list[str],
    actor_id: str = "merchant_demo",
) -> tuple[EvidenceSubmission, dict]:
    try:
        validated = EvidenceSubmissionRequest(
            merchant_explanation_text=merchant_explanation_text,
            evidence_references=evidence_references,
        )
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    case = get_case(session, case_id)
    if case is None:
        raise CaseNotFoundError(f"No case found for case_id={case_id}")
    if case.case_status != "EVIDENCE_REQUESTED":
        raise InvalidTransitionError("Evidence can only be submitted while a case is in EVIDENCE_REQUESTED status.")

    evidence = EvidenceSubmission(
        evidence_id=f"evidence_{uuid.uuid4().hex[:16]}",
        case_id=case_id,
        submitted_by_actor_type="merchant_demo",
        merchant_explanation_text=validated.merchant_explanation_text,
        evidence_references_json=validated.evidence_references,
        status="SUBMITTED",
    )
    add_evidence(session, evidence)

    case.case_status = "EVIDENCE_SUBMITTED"
    session.flush()

    audit_event = record_event(session, case_id, "merchant_demo", actor_id, "EVIDENCE_SUBMITTED", {
        "evidence_id": evidence.evidence_id,
        "evidence_references": validated.evidence_references,
        "explanation_length": len(validated.merchant_explanation_text),
    })
    return evidence, audit_event
