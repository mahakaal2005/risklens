"""Shared API response/request schemas for the Milestone 6 FastAPI layer.

Every response includes a synthetic_data_notice. No schema here ever
includes label_high_loss_next_30d, latent_state_for_demo_only, a
diagnostic-only feature, a raw model coefficient, or any
financial-enforcement field -- this is a local synthetic-data
demonstration only.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

from pydantic import BaseModel

from app.schemas.cases import CaseStatus, FinalOutcome, ReviewerAction

SYNTHETIC_DATA_NOTICE = "Local synthetic-data demonstration only."


class ActorType(str, Enum):
    SYSTEM = "system"
    ANALYST_DEMO = "analyst_demo"
    MERCHANT_DEMO = "merchant_demo"


class AuditEventType(str, Enum):
    ASSESSMENT_GENERATED = "ASSESSMENT_GENERATED"
    EXPLANATION_GENERATED = "EXPLANATION_GENERATED"
    REVIEW_CASE_CREATED = "REVIEW_CASE_CREATED"
    REVIEW_CASE_RECOMMENDED = "REVIEW_CASE_RECOMMENDED"
    EVIDENCE_REQUEST_RECOMMENDED = "EVIDENCE_REQUEST_RECOMMENDED"
    MANUAL_REVIEW_RECOMMENDED = "MANUAL_REVIEW_RECOMMENDED"
    EVIDENCE_SUBMITTED = "EVIDENCE_SUBMITTED"
    REVIEW_STARTED = "REVIEW_STARTED"
    CASE_CLEARED = "CASE_CLEARED"
    CASE_MARKED_FALSE_POSITIVE = "CASE_MARKED_FALSE_POSITIVE"
    CASE_MARKED_OPERATIONAL_ISSUE = "CASE_MARKED_OPERATIONAL_ISSUE"
    CASE_MARKED_INCONCLUSIVE = "CASE_MARKED_INCONCLUSIVE"
    CASE_ESCALATED = "CASE_ESCALATED"


class ReviewActionAPI(str, Enum):
    """Superset of app.schemas.cases.ReviewerAction, adding START_REVIEW --
    the "reviewer begins review" lifecycle step, which the service layer
    implements as case_service.start_review() rather than a
    VALID_TRANSITIONS-table reviewer action."""

    CLEAR_CASE = "CLEAR_CASE"
    MARK_FALSE_POSITIVE = "MARK_FALSE_POSITIVE"
    REQUEST_EVIDENCE = "REQUEST_EVIDENCE"
    MARK_OPERATIONAL_ISSUE = "MARK_OPERATIONAL_ISSUE"
    ESCALATE_CASE = "ESCALATE_CASE"
    MARK_INCONCLUSIVE = "MARK_INCONCLUSIVE"
    START_REVIEW = "START_REVIEW"


class ErrorCode(str, Enum):
    CASE_NOT_FOUND = "CASE_NOT_FOUND"
    INVALID_CASE_TRANSITION = "INVALID_CASE_TRANSITION"
    INVALID_EVIDENCE_REFERENCE = "INVALID_EVIDENCE_REFERENCE"
    INVALID_REVIEW_ACTION = "INVALID_REVIEW_ACTION"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    METRICS_NOT_AVAILABLE = "METRICS_NOT_AVAILABLE"
    METRICS_ARTIFACT_INVALID = "METRICS_ARTIFACT_INVALID"
    INTERNAL_SAFE_ERROR = "INTERNAL_SAFE_ERROR"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    synthetic_data_notice: str = SYNTHETIC_DATA_NOTICE


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "clearrisk-recover"
    environment: str = "local-demo"
    data_mode: str = "synthetic-only"
    payment_actions_enabled: bool = False


class CaseSummary(BaseModel):
    case_id: str
    merchant_id: str
    week_start: str
    case_status: CaseStatus
    risk_signal_intensity: str
    recommendation: str
    created_at: dt.datetime
    updated_at: dt.datetime
    final_outcome: FinalOutcome | None = None
    synthetic_data_notice: str = SYNTHETIC_DATA_NOTICE


class CaseListResponse(BaseModel):
    items: list[CaseSummary]
    limit: int
    offset: int
    total: int
    synthetic_data_notice: str = SYNTHETIC_DATA_NOTICE


class EvidenceSubmissionSummary(BaseModel):
    evidence_id: str
    submitted_at: dt.datetime
    status: str
    evidence_references: list[str]


class CaseDetailResponse(BaseModel):
    case_id: str
    merchant_id: str
    week_start: str
    case_status: CaseStatus
    risk_signal_intensity: str
    recommendation: str
    policy_explanation: str
    model_probability: float | None
    rules_only_score: int
    analyst_summary: str
    merchant_safe_explanation: dict
    triggered_rules: list[str]
    evidence_checklist: list[str]
    model_version: str | None
    rules_version: str
    created_at: dt.datetime
    updated_at: dt.datetime
    resolved_at: dt.datetime | None
    final_outcome: FinalOutcome | None
    reviewer_note: str | None
    evidence_submissions: list[EvidenceSubmissionSummary]
    synthetic_data_notice: str = SYNTHETIC_DATA_NOTICE


class AuditEventResponse(BaseModel):
    event_sequence_number: int
    event_timestamp: dt.datetime
    actor_type: str
    actor_id: str
    event_type: str
    event_payload: dict


class AuditTimelineResponse(BaseModel):
    case_id: str
    events: list[AuditEventResponse]
    synthetic_data_notice: str = SYNTHETIC_DATA_NOTICE


class ReviewActionRequest(BaseModel):
    """No actor-id field: the reviewer's identity is derived entirely from
    the authenticated session (Phase 2 auth), never from client input --
    see docs/PHASE_2_AUTH_DESIGN.md Section 6."""

    action: ReviewActionAPI
    reviewer_note: str


class ReviewActionResponse(BaseModel):
    case: CaseDetailResponse
    new_audit_events: list[AuditEventResponse]
    synthetic_data_notice: str = SYNTHETIC_DATA_NOTICE


class EvidenceSubmissionRequestBody(BaseModel):
    """No actor-id field: the merchant's identity is derived entirely from
    the authenticated session (Phase 2 auth), never from client input --
    see docs/PHASE_2_AUTH_DESIGN.md Section 6."""

    merchant_explanation_text: str
    evidence_references: list[str] = []


class EvidenceSubmissionResponse(BaseModel):
    evidence_id: str
    case_id: str
    case_status: CaseStatus
    submitted_at: dt.datetime
    evidence_references: list[str]
    new_audit_event: AuditEventResponse
    synthetic_data_notice: str = SYNTHETIC_DATA_NOTICE


class MetricsResponse(BaseModel):
    status: str  # "available" | "not_available"
    data_mode: str = "synthetic-only"
    message: str
    error_code: str | None = None  # "METRICS_NOT_AVAILABLE" | "METRICS_ARTIFACT_INVALID", when status is not_available
    generation_command: str | None = None  # e.g. "python3 -m ml.evaluate_model", when status is not_available
    dataset_seed: int | None = None
    dataset_version: str | None = None
    held_out_test_date_range: dict | None = None
    selected_threshold: float | None = None
    rules_only_metrics: dict | None = None
    logistic_regression_metrics: dict | None = None
    combined_policy_metrics: dict | None = None
    near_perfect_investigation_status: str | None = None
    limitation: str = (
        "Synthetic-data metrics demonstrate prototype workflow only and do not prove "
        "real-world chargeback-risk performance."
    )
    synthetic_data_notice: str = SYNTHETIC_DATA_NOTICE
