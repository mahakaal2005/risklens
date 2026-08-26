"""Append-only application audit log service.

This module exposes only a record() function and read functions -- there is
no update or delete for an audit event anywhere in this codebase. This is an
application-level append-only log, not a cryptographically immutable or
WORM (write-once-read-many) store; see SECURITY.md and
docs/AUDIT_EVENT_SCHEMA.md.
"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.db.repositories import append_audit_event, get_audit_events_for_case
from app.schemas.audit import FORBIDDEN_PAYLOAD_TERMS
from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN


class UnsafeAuditPayloadError(Exception):
    pass


def _assert_payload_is_safe(payload: dict) -> None:
    serialized = json.dumps(payload).lower()
    if LABEL_COLUMN in serialized or LATENT_STATE_COLUMN in serialized:
        raise UnsafeAuditPayloadError("Audit payload must never reference the target label or latent state.")
    for term in FORBIDDEN_PAYLOAD_TERMS:
        if term in serialized:
            raise UnsafeAuditPayloadError(f"Audit payload must never contain enforcement term: {term!r}")


def record_event(session: Session, case_id: str, actor_type: str, actor_id: str, event_type: str, payload: dict) -> dict:
    _assert_payload_is_safe(payload)
    event = append_audit_event(session, case_id, actor_type, actor_id, event_type, payload)
    return {
        "audit_event_id": event.audit_event_id,
        "case_id": event.case_id,
        "event_timestamp": event.event_timestamp,
        "actor_type": event.actor_type,
        "actor_id": event.actor_id,
        "event_type": event.event_type,
        "event_payload_json": event.event_payload_json,
        "event_sequence_number": event.event_sequence_number,
    }


def get_case_timeline(session: Session, case_id: str) -> list[dict]:
    """Ordered by sequence number then timestamp -- read-only."""
    events = get_audit_events_for_case(session, case_id)
    return [
        {
            "audit_event_id": e.audit_event_id,
            "case_id": e.case_id,
            "event_timestamp": e.event_timestamp,
            "actor_type": e.actor_type,
            "actor_id": e.actor_id,
            "event_type": e.event_type,
            "event_payload_json": e.event_payload_json,
            "event_sequence_number": e.event_sequence_number,
        }
        for e in events
    ]
