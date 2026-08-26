"""Pydantic read schema for append-only audit events.

No update/delete schema exists here by design -- the audit log is
append-only at the application layer (see docs/AUDIT_EVENT_SCHEMA.md and
SECURITY.md for the append-only-vs-cryptographic-immutability distinction).
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel

REQUIRED_EVENT_TYPES = {
    "ASSESSMENT_GENERATED",
    "EXPLANATION_GENERATED",
    "REVIEW_CASE_CREATED",
    "REVIEW_CASE_RECOMMENDED",
    "EVIDENCE_REQUEST_RECOMMENDED",
    "MANUAL_REVIEW_RECOMMENDED",
    "EVIDENCE_SUBMITTED",
    "REVIEW_STARTED",
    "CASE_CLEARED",
    "CASE_MARKED_FALSE_POSITIVE",
    "CASE_MARKED_OPERATIONAL_ISSUE",
    "CASE_MARKED_INCONCLUSIVE",
    "CASE_ESCALATED",
}

FORBIDDEN_PAYLOAD_TERMS = ["freeze", "ban", "terminate", "hold settlement", "reject payment"]


class AuditEventRead(BaseModel):
    audit_event_id: str
    case_id: str
    event_timestamp: dt.datetime
    actor_type: str
    actor_id: str
    event_type: str
    event_payload_json: dict
    event_sequence_number: int

    model_config = {"from_attributes": True}
