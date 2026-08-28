"""Reviewer-feedback label overrides for the manually-triggered retraining
loop (Phase 2). See docs/PHASE_2_FEEDBACK_RETRAINING_DESIGN.md.

Only FALSE_POSITIVE and CONFIRMED_RISK resolutions map to a label
correction -- OPERATIONAL_ISSUE and INCONCLUSIVE are excluded entirely,
since their correct label is a genuine judgment call this project does
not make silently (confirmed with the user before implementation).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ReviewCase

LABEL_BY_FINAL_OUTCOME = {
    "FALSE_POSITIVE": 0,
    "CONFIRMED_RISK": 1,
}


def get_feedback_label_overrides(session: Session) -> list[dict]:
    """Returns one entry per resolved case whose final_outcome is
    FALSE_POSITIVE or CONFIRMED_RISK, with the corresponding corrected
    label. Does not touch the synthetic dataset or the model -- this is
    read-only export logic; ml/retrain_with_feedback.py applies it."""
    stmt = select(ReviewCase).where(ReviewCase.final_outcome.in_(list(LABEL_BY_FINAL_OUTCOME)))
    cases = session.execute(stmt).scalars().all()
    return [
        {
            "case_id": case.case_id,
            "merchant_id": case.merchant_id,
            "week_start": case.week_start,
            "final_outcome": case.final_outcome,
            "corrected_label": LABEL_BY_FINAL_OUTCOME[case.final_outcome],
        }
        for case in cases
    ]
