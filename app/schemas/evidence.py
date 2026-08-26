"""Pydantic schemas for simulated merchant evidence submission.

Evidence references are simulated strings only (e.g. invoice_demo_001.pdf).
No file upload, external URL retrieval, or document storage exists here or
anywhere in this codebase.
"""

from __future__ import annotations

import datetime as dt
import re

from pydantic import BaseModel, field_validator

MAX_EVIDENCE_REFERENCES = 5
MAX_REFERENCE_LENGTH = 100

# Safe demo filename/identifier pattern only: letters, digits, underscore,
# hyphen, and a single dot before a short extension. No slashes, no "://",
# no shell metacharacters.
SAFE_REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+(\.[A-Za-z0-9]{1,10})?$")

FORBIDDEN_SUBSTRINGS = ["://", "..", "/", "\\", ";", "|", "&", "$", "`", "\n", "\r"]


def validate_evidence_reference(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("Evidence reference must not be blank.")
    if len(value) > MAX_REFERENCE_LENGTH:
        raise ValueError(f"Evidence reference exceeds {MAX_REFERENCE_LENGTH} characters.")
    for forbidden in FORBIDDEN_SUBSTRINGS:
        if forbidden in value:
            raise ValueError(f"Evidence reference contains a disallowed sequence: {forbidden!r}")
    if not SAFE_REFERENCE_PATTERN.match(value):
        raise ValueError("Evidence reference must be a safe demo filename or identifier only.")
    return value


class EvidenceSubmissionRequest(BaseModel):
    merchant_explanation_text: str
    evidence_references: list[str] = []

    @field_validator("merchant_explanation_text")
    @classmethod
    def explanation_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Merchant explanation text must not be empty.")
        return v

    @field_validator("evidence_references")
    @classmethod
    def references_must_be_safe_and_bounded(cls, v: list[str]) -> list[str]:
        if len(v) > MAX_EVIDENCE_REFERENCES:
            raise ValueError(f"At most {MAX_EVIDENCE_REFERENCES} evidence references are allowed.")
        return [validate_evidence_reference(item) for item in v]


class EvidenceSubmissionRead(BaseModel):
    evidence_id: str
    case_id: str
    submitted_by_actor_type: str
    merchant_explanation_text: str
    evidence_references_json: list[str]
    submitted_at: dt.datetime
    status: str

    model_config = {"from_attributes": True}
