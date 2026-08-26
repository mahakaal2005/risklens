import json

import joblib
import pandas as pd
import pytest

from app.db.database import create_db_engine, init_db, make_session_factory, session_scope
from app.db.repositories import get_audit_events_for_case, get_case
from app.services.case_service import create_case_from_packet
from ml.explain_cases import DIAGNOSTIC_ONLY_FEATURES, compute_top_factors
from ml.features import compute_features
from ml.model_utils import ML_FEATURE_COLUMNS

ARTIFACT_PATH = "ml/artifacts/logistic_regression_v0.1.0.joblib"

HIGH_RISK_ROW = dict(
    merchant_id="merchant_demo_0001", week_start="2025-11-17", merchant_category="apparel",
    merchant_age_days=400, transaction_count_30d=500, transaction_volume_30d=500000.0,
    transaction_volume_previous_30d=480000.0, refund_count_30d=8,
    top_dispute_reason_category="other", previous_review_outcome="none",
    refund_rate_30d=0.07, refund_rate_previous_30d=0.02,
    chargeback_rate_30d=0.03, chargeback_rate_previous_30d=0.01,
    delivery_evidence_coverage=0.30, support_ticket_rate=0.07,
    average_support_resolution_time_hours=70,
)


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(ARTIFACT_PATH)


@pytest.fixture()
def session_factory():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    return make_session_factory(engine)


@pytest.fixture(scope="module")
def demo_packets():
    with open("demo_data/demo_case_packets.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_support_ticket_rate_is_configured_diagnostic_only():
    assert "support_ticket_rate" in DIAGNOSTIC_ONLY_FEATURES


def test_diagnostic_only_feature_excluded_from_top_factors(pipeline):
    features = compute_features(HIGH_RISK_ROW)
    row_df = pd.DataFrame([features])[ML_FEATURE_COLUMNS]
    factors = compute_top_factors(pipeline, row_df)
    feature_names = {f["feature"] for f in factors}
    assert "support_ticket_rate" not in feature_names


def test_diagnostic_only_feature_absent_from_demo_packet_explanations(demo_packets):
    """support_ticket_rate must never appear as a ranked, causal-sounding
    model factor (top_model_factors) or in the merchant-safe explanation.
    It IS allowed to appear as a raw trend value (analyst_explanation.
    trend_values) and inside a triggered rule's own transparent explanation
    text (e.g. SUPPORT_OPERATIONAL_STRESS) -- those are documented,
    validated, rule-based uses, not an unvalidated model-coefficient claim.
    """
    for name, packet in demo_packets.items():
        top_factors_text = json.dumps(packet["analyst_explanation"]["top_model_factors"])
        merchant_text = json.dumps(packet["merchant_safe_explanation"])
        assert "support_ticket_rate" not in top_factors_text, f"{name} leaked support_ticket_rate in top_model_factors"
        assert "support_ticket_rate" not in merchant_text, f"{name} leaked support_ticket_rate in merchant-safe explanation"


def test_diagnostic_only_feature_absent_from_persisted_case_and_audit_payloads(session_factory, demo_packets):
    for name, packet in demo_packets.items():
        with session_scope(session_factory) as session:
            case, events = create_case_from_packet(session, packet)
            if case is None:
                continue
            case_id = case.case_id

        with session_scope(session_factory) as session:
            stored = get_case(session, case_id)
            serialized_case = json.dumps({
                "analyst_summary": stored.analyst_summary,
                "merchant_safe_explanation": stored.merchant_safe_explanation,
            })
            assert "support_ticket_rate" not in serialized_case, f"{name} persisted support_ticket_rate in case fields"

            stored_events = get_audit_events_for_case(session, case_id)
            for e in stored_events:
                assert "support_ticket_rate" not in json.dumps(e.event_payload_json), f"{name} leaked support_ticket_rate in audit payload"


def test_audit_payloads_never_contain_forbidden_enforcement_terms(session_factory, demo_packets):
    forbidden_terms = ["freeze", "ban", "terminate", "hold settlement", "reject payment"]
    for name, packet in demo_packets.items():
        with session_scope(session_factory) as session:
            case, events = create_case_from_packet(session, packet)
            if case is None:
                continue
            case_id = case.case_id

        with session_scope(session_factory) as session:
            stored_events = get_audit_events_for_case(session, case_id)
            for e in stored_events:
                payload_text = json.dumps(e.event_payload_json).lower()
                for term in forbidden_terms:
                    assert term not in payload_text, f"{name} audit event {e.event_type} contains forbidden term {term!r}"
