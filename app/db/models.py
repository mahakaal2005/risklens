"""SQLAlchemy ORM models for RiskLens's case workflow.

No field here ever stores label_high_loss_next_30d or
latent_state_for_demo_only -- the service layer (app/services/case_service.py)
is responsible for only ever passing safe case-packet fields in, and this is
verified by tests/test_explanation_safety.py and tests/test_case_service.py.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.time_utils import utcnow as _utcnow


class Base(DeclarativeBase):
    pass


class User(Base):
    """Local demo accounts only -- see docs/PHASE_2_AUTH_DESIGN.md. Not real
    people; seeded by scripts/seed_demo_users.py. Password hashing is a
    stdlib pbkdf2_hmac KDF, explicitly documented as adequate for this local
    prototype and not a claim of production password-security compliance.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    username: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    password_salt: Mapped[str] = mapped_column(String, nullable=False)

    role: Mapped[str] = mapped_column(String, nullable=False)  # "reviewer" | "merchant" | "risk_manager"
    actor_id: Mapped[str] = mapped_column(String, nullable=False)  # recorded in audit events
    merchant_id: Mapped[str | None] = mapped_column(String, nullable=True)  # only set for role="merchant"
    display_name: Mapped[str] = mapped_column(String, nullable=False)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    sessions: Mapped[list["UserSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class UserSession(Base):
    """Opaque server-side session token, not a JWT -- simpler and directly
    revocable for a local single-operator demo. See
    docs/PHASE_2_AUTH_DESIGN.md Section 4."""

    __tablename__ = "sessions"

    token: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user: Mapped["User"] = relationship(back_populates="sessions")


class ReviewCase(Base):
    __tablename__ = "review_cases"

    case_id: Mapped[str] = mapped_column(String, primary_key=True)
    merchant_id: Mapped[str] = mapped_column(String, nullable=False)
    week_start: Mapped[str] = mapped_column(String, nullable=False)

    case_status: Mapped[str] = mapped_column(String, nullable=False, default="OPEN")

    risk_signal_intensity: Mapped[str] = mapped_column(String, nullable=False)
    model_probability: Mapped[float | None] = mapped_column(nullable=True)
    selected_threshold: Mapped[float] = mapped_column(nullable=False)
    rules_only_score: Mapped[int] = mapped_column(Integer, nullable=False)
    recommendation: Mapped[str] = mapped_column(String, nullable=False)
    policy_explanation: Mapped[str] = mapped_column(Text, nullable=False)

    analyst_summary: Mapped[str] = mapped_column(Text, nullable=False)
    merchant_safe_explanation: Mapped[dict] = mapped_column(JSON, nullable=False)
    triggered_rules_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    triggered_rule_explanations_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    evidence_checklist_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    rules_version: Mapped[str] = mapped_column(String, nullable=False)
    synthetic_data_notice: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    resolved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    final_outcome: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer_actor: Mapped[str | None] = mapped_column(String, nullable=True)
    decision_timestamp: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    evidence_submissions: Mapped[list["EvidenceSubmission"]] = relationship(back_populates="case", cascade="all, delete-orphan")
    audit_events: Mapped[list["AuditEvent"]] = relationship(back_populates="case", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("merchant_id", "week_start", name="uq_case_merchant_week"),)


class EvidenceSubmission(Base):
    __tablename__ = "evidence_submissions"

    evidence_id: Mapped[str] = mapped_column(String, primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("review_cases.case_id"), nullable=False)

    submitted_by_actor_type: Mapped[str] = mapped_column(String, nullable=False, default="merchant_demo")
    merchant_explanation_text: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_references_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    submitted_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    status: Mapped[str] = mapped_column(String, nullable=False, default="SUBMITTED")

    case: Mapped["ReviewCase"] = relationship(back_populates="evidence_submissions")
    attachments: Mapped[list["EvidenceAttachment"]] = relationship(back_populates="evidence", cascade="all, delete-orphan")


class EvidenceAttachment(Base):
    """A real uploaded file attached to an evidence submission (Phase 2 --
    see docs/PHASE_2_EVIDENCE_ATTACHMENTS_DESIGN.md). stored_filename is a
    server-generated name; the client-supplied original_filename is kept
    only for display, never used to construct a filesystem path."""

    __tablename__ = "evidence_attachments"

    attachment_id: Mapped[str] = mapped_column(String, primary_key=True)
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence_submissions.evidence_id"), nullable=False)

    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    stored_filename: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    evidence: Mapped["EvidenceSubmission"] = relationship(back_populates="attachments")


class AuditEvent(Base):
    __tablename__ = "audit_events"

    audit_event_id: Mapped[str] = mapped_column(String, primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("review_cases.case_id"), nullable=False)

    event_timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    actor_type: Mapped[str] = mapped_column(String, nullable=False)
    actor_id: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    event_payload_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    event_sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)

    case: Mapped["ReviewCase"] = relationship(back_populates="audit_events")

    __table_args__ = (UniqueConstraint("case_id", "event_sequence_number", name="uq_audit_case_sequence"),)
