"""Tests for app/services/feedback_service.py -- exporting reviewer
resolutions as training-label overrides. See
docs/PHASE_2_FEEDBACK_RETRAINING_DESIGN.md."""

from __future__ import annotations

import pytest

from app.db.database import create_db_engine, init_db, make_session_factory
from app.db.models import ReviewCase
from app.services.feedback_service import get_feedback_label_overrides


@pytest.fixture()
def db_session(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'test_feedback_service.db'}")
    init_db(engine)
    factory = make_session_factory(engine)
    session = factory()
    yield session
    session.close()


def _make_case(case_id, merchant_id, week_start, final_outcome) -> ReviewCase:
    return ReviewCase(
        case_id=case_id,
        merchant_id=merchant_id,
        week_start=week_start,
        case_status="RESOLVED" if final_outcome else "OPEN",
        risk_signal_intensity="Medium",
        selected_threshold=0.1,
        rules_only_score=30,
        recommendation="REQUEST_EVIDENCE",
        policy_explanation="x",
        analyst_summary="x",
        merchant_safe_explanation={},
        rules_version="1.0",
        synthetic_data_notice="x",
        final_outcome=final_outcome,
    )


def test_only_false_positive_and_confirmed_risk_are_exported(db_session):
    db_session.add_all([
        _make_case("c1", "merchant_demo_0001", "2025-01-06", "FALSE_POSITIVE"),
        _make_case("c2", "merchant_demo_0002", "2025-01-13", "CONFIRMED_RISK"),
        _make_case("c3", "merchant_demo_0003", "2025-01-20", "OPERATIONAL_ISSUE"),
        _make_case("c4", "merchant_demo_0004", "2025-01-27", "INCONCLUSIVE"),
        _make_case("c5", "merchant_demo_0005", "2025-02-03", None),
    ])
    db_session.commit()

    overrides = get_feedback_label_overrides(db_session)
    case_ids = {o["case_id"] for o in overrides}
    assert case_ids == {"c1", "c2"}


def test_false_positive_maps_to_label_zero(db_session):
    db_session.add(_make_case("c1", "merchant_demo_0001", "2025-01-06", "FALSE_POSITIVE"))
    db_session.commit()
    overrides = get_feedback_label_overrides(db_session)
    assert overrides[0]["corrected_label"] == 0


def test_confirmed_risk_maps_to_label_one(db_session):
    db_session.add(_make_case("c1", "merchant_demo_0001", "2025-01-06", "CONFIRMED_RISK"))
    db_session.commit()
    overrides = get_feedback_label_overrides(db_session)
    assert overrides[0]["corrected_label"] == 1


def test_no_resolved_cases_returns_empty_list(db_session):
    db_session.add(_make_case("c1", "merchant_demo_0001", "2025-01-06", None))
    db_session.commit()
    assert get_feedback_label_overrides(db_session) == []


def test_override_includes_merchant_id_and_week_start_for_joining(db_session):
    db_session.add(_make_case("c1", "merchant_demo_0007", "2025-03-10", "FALSE_POSITIVE"))
    db_session.commit()
    overrides = get_feedback_label_overrides(db_session)
    assert overrides[0]["merchant_id"] == "merchant_demo_0007"
    assert overrides[0]["week_start"] == "2025-03-10"
