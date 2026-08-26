"""Thin data-access layer over the SQLAlchemy models.

Deliberately exposes no update/delete methods for AuditEvent -- the audit
log is append-only at the application layer, enforced by only ever
providing a create + read interface here.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import AuditEvent, EvidenceSubmission, ReviewCase


def get_case(session: Session, case_id: str) -> ReviewCase | None:
    return session.get(ReviewCase, case_id)


def get_case_by_merchant_week(session: Session, merchant_id: str, week_start: str) -> ReviewCase | None:
    stmt = select(ReviewCase).where(ReviewCase.merchant_id == merchant_id, ReviewCase.week_start == week_start)
    return session.execute(stmt).scalar_one_or_none()


def list_cases(
    session: Session,
    status: str | None = None,
    recommendation: str | None = None,
    intensity: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[ReviewCase], int]:
    """Read-only, filtered, paginated case listing. Pure data access -- no
    business/workflow logic (that stays in app/services/case_service.py)."""
    filters = []
    if status is not None:
        filters.append(ReviewCase.case_status == status)
    if recommendation is not None:
        filters.append(ReviewCase.recommendation == recommendation)
    if intensity is not None:
        filters.append(ReviewCase.risk_signal_intensity == intensity)

    count_stmt = select(func.count()).select_from(ReviewCase)
    if filters:
        count_stmt = count_stmt.where(*filters)
    total = session.execute(count_stmt).scalar_one()

    stmt = select(ReviewCase)
    if filters:
        stmt = stmt.where(*filters)
    stmt = stmt.order_by(ReviewCase.created_at.desc(), ReviewCase.case_id.asc()).limit(limit).offset(offset)
    items = list(session.execute(stmt).scalars().all())
    return items, total


def add_case(session: Session, case: ReviewCase) -> ReviewCase:
    session.add(case)
    session.flush()
    return case


def add_evidence(session: Session, evidence: EvidenceSubmission) -> EvidenceSubmission:
    session.add(evidence)
    session.flush()
    return evidence


def get_next_sequence_number(session: Session, case_id: str) -> int:
    stmt = select(AuditEvent).where(AuditEvent.case_id == case_id).order_by(AuditEvent.event_sequence_number.desc())
    last_event = session.execute(stmt).scalars().first()
    return (last_event.event_sequence_number + 1) if last_event else 1


def append_audit_event(
    session: Session,
    case_id: str,
    actor_type: str,
    actor_id: str,
    event_type: str,
    event_payload: dict,
) -> AuditEvent:
    """The only way to write an AuditEvent. There is no update/delete
    counterpart anywhere in this module."""
    event = AuditEvent(
        audit_event_id=f"audit_{uuid.uuid4().hex[:16]}",
        case_id=case_id,
        actor_type=actor_type,
        actor_id=actor_id,
        event_type=event_type,
        event_payload_json=event_payload,
        event_sequence_number=get_next_sequence_number(session, case_id),
    )
    session.add(event)
    session.flush()
    return event


def get_audit_events_for_case(session: Session, case_id: str) -> list[AuditEvent]:
    stmt = (
        select(AuditEvent)
        .where(AuditEvent.case_id == case_id)
        .order_by(AuditEvent.event_sequence_number.asc(), AuditEvent.event_timestamp.asc())
    )
    return list(session.execute(stmt).scalars().all())
