import json

import joblib
import pytest

from ml.case_packet import build_case_packet
from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN
from ml.model_utils import ALLOWED_RECOMMENDATIONS, combined_policy
from ml.rules_engine import score_merchant_week

ARTIFACT_PATH = "ml/artifacts/logistic_regression_v0.1.0.joblib"

BASE_ROW = dict(
    merchant_id="merchant_demo_0001", week_start="2025-11-17", merchant_category="apparel",
    merchant_age_days=400, transaction_count_30d=500, transaction_volume_30d=500000.0,
    transaction_volume_previous_30d=480000.0, refund_count_30d=8,
    top_dispute_reason_category="other", previous_review_outcome="none",
)

STABLE_ROW = {**BASE_ROW, "refund_rate_30d": 0.015, "refund_rate_previous_30d": 0.014,
              "chargeback_rate_30d": 0.003, "chargeback_rate_previous_30d": 0.0028,
              "delivery_evidence_coverage": 0.92, "support_ticket_rate": 0.01,
              "average_support_resolution_time_hours": 20}

REFUND_ONLY_ROW = {**BASE_ROW, "refund_rate_30d": 0.06, "refund_rate_previous_30d": 0.02,
                    "chargeback_rate_30d": 0.004, "chargeback_rate_previous_30d": 0.0035,
                    "delivery_evidence_coverage": 0.88, "support_ticket_rate": 0.02,
                    "average_support_resolution_time_hours": 24}

EVIDENCE_GAP_ROW = {**BASE_ROW, "refund_rate_30d": 0.05, "refund_rate_previous_30d": 0.025,
                     "chargeback_rate_30d": 0.012, "chargeback_rate_previous_30d": 0.009,
                     "delivery_evidence_coverage": 0.60, "support_ticket_rate": 0.045,
                     "average_support_resolution_time_hours": 55}

HIGH_RISK_ROW = {**BASE_ROW, "refund_rate_30d": 0.07, "refund_rate_previous_30d": 0.02,
                  "chargeback_rate_30d": 0.03, "chargeback_rate_previous_30d": 0.01,
                  "delivery_evidence_coverage": 0.30, "support_ticket_rate": 0.07,
                  "average_support_resolution_time_hours": 70}


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(ARTIFACT_PATH)


def _build(row, pipeline, ml_probability=0.5, threshold=0.1):
    rules_result = score_merchant_week(row)
    decision = combined_policy(ml_probability, threshold, set(rules_result["triggered_rules"]))
    return build_case_packet(
        record=row, rules_result=rules_result, ml_probability=ml_probability,
        selected_threshold=threshold, model_version="0.1.0", rules_version="0.2.0",
        combined_decision=decision, pipeline=pipeline,
    )


def test_packet_excludes_label_and_latent_state_completely(pipeline):
    row = {**HIGH_RISK_ROW, LABEL_COLUMN: 1, LATENT_STATE_COLUMN: "high_risk_merchant_behaviour"}
    packet = _build(row, pipeline)
    serialized = json.dumps(packet)
    assert LABEL_COLUMN not in serialized
    assert LATENT_STATE_COLUMN not in serialized
    assert "high_risk_merchant_behaviour" not in serialized


def test_merchant_safe_explanation_forbidden_terms_absent(pipeline):
    packet = _build(HIGH_RISK_ROW, pipeline)
    merchant_text = json.dumps(packet["merchant_safe_explanation"]).lower()
    for term in ["fraud confirmed", "ban", "freeze", "terminate", "latent"]:
        assert term not in merchant_text
    assert str(packet["assessment"]["model_probability"]) not in merchant_text
    assert "selected_threshold" not in packet["merchant_safe_explanation"]
    assert "model_probability" not in packet["merchant_safe_explanation"]


def test_analyst_explanation_trend_values_only_reference_prediction_time_fields(pipeline):
    packet = _build(HIGH_RISK_ROW, pipeline)
    trends = packet["analyst_explanation"]["trend_values"]
    assert set(trends.keys()) == {
        "refund_rate_current_vs_prior", "chargeback_rate_current_vs_prior",
        "transaction_volume_current_vs_prior", "delivery_evidence_coverage",
        "support_ticket_rate", "support_resolution_time_hours",
    }


def test_degraded_mode_without_model_artifact():
    rules_result = score_merchant_week(HIGH_RISK_ROW)
    decision = combined_policy(0.0, 0.1, set(rules_result["triggered_rules"]))
    packet = build_case_packet(
        record=HIGH_RISK_ROW, rules_result=rules_result, ml_probability=None,
        selected_threshold=0.1, model_version=None, rules_version="0.2.0",
        combined_decision=decision, pipeline=None,
    )
    assert packet["assessment"]["degraded_mode"] is True
    assert packet["assessment"]["model_probability"] is None
    assert packet["analyst_explanation"]["top_model_factors"] == []
    assert "degraded" in packet["analyst_explanation"]["uncertainty_statement"].lower() or "rules engine only" in packet["analyst_explanation"]["uncertainty_statement"]


@pytest.mark.parametrize("row", [STABLE_ROW, REFUND_ONLY_ROW, EVIDENCE_GAP_ROW, HIGH_RISK_ROW])
def test_recommendation_is_always_in_allowed_enum(pipeline, row):
    packet = _build(row, pipeline)
    assert packet["assessment"]["recommendation"] in ALLOWED_RECOMMENDATIONS


def test_evidence_checklist_refund_only_case_excludes_delivery_proof(pipeline):
    packet = _build(REFUND_ONLY_ROW, pipeline)
    checklist_text = " ".join(packet["evidence_checklist"]).lower()
    assert "delivery" not in checklist_text
    assert "refund" in checklist_text


def test_evidence_checklist_evidence_gap_case_includes_fulfilment_proof(pipeline):
    packet = _build(EVIDENCE_GAP_ROW, pipeline)
    checklist_text = " ".join(packet["evidence_checklist"]).lower()
    assert "fulfilment" in checklist_text or "delivery" in checklist_text


def test_evidence_checklist_support_stress_case_includes_support_records(pipeline):
    packet = _build(EVIDENCE_GAP_ROW, pipeline)
    checklist_text = " ".join(packet["evidence_checklist"]).lower()
    assert "support" in checklist_text


def test_stable_case_has_empty_evidence_checklist(pipeline):
    packet = _build(STABLE_ROW, pipeline, ml_probability=0.01)
    assert packet["evidence_checklist"] == []


def test_audit_preview_events_match_recommendation_approve(pipeline):
    packet = _build(STABLE_ROW, pipeline, ml_probability=0.01)
    event_types = {e["event_type"] for e in packet["audit_preview_events"]}
    assert event_types == {"ASSESSMENT_GENERATED", "EXPLANATION_GENERATED"}
    assert all(e["preview_only"] is True for e in packet["audit_preview_events"])


def test_audit_preview_events_match_recommendation_manual_review(pipeline):
    packet = _build(HIGH_RISK_ROW, pipeline, ml_probability=0.9)
    event_types = {e["event_type"] for e in packet["audit_preview_events"]}
    assert "MANUAL_REVIEW_RECOMMENDED" in event_types
    assert "REVIEW_CASE_RECOMMENDED" in event_types


def test_packet_json_serialization_is_stable_and_valid(pipeline):
    packet = _build(HIGH_RISK_ROW, pipeline)
    serialized_a = json.dumps(packet, sort_keys=True)
    reloaded = json.loads(serialized_a)
    serialized_b = json.dumps(reloaded, sort_keys=True)
    assert serialized_a == serialized_b


def test_five_demo_packet_types_are_generated(tmp_path):
    from ml.generate_demo_cases import generate_demo_cases

    output_path = tmp_path / "demo_case_packets.json"
    packets = generate_demo_cases(output_path=output_path)

    expected_types = {
        "stable_merchant", "seasonal_sale_false_positive_candidate",
        "operational_fulfilment_problem", "high_risk_combined_loss_case",
        "early_hidden_risk_case",
    }
    assert set(packets.keys()) == expected_types
    assert output_path.exists()


def test_explanations_are_deterministic_for_same_inputs(pipeline):
    packet_a = _build(HIGH_RISK_ROW, pipeline)
    packet_b = _build(HIGH_RISK_ROW, pipeline)
    assert packet_a["analyst_explanation"]["summary"] == packet_b["analyst_explanation"]["summary"]
    assert packet_a["analyst_explanation"]["top_model_factors"] == packet_b["analyst_explanation"]["top_model_factors"]
    assert packet_a["merchant_safe_explanation"] == packet_b["merchant_safe_explanation"]
    assert packet_a["evidence_checklist"] == packet_b["evidence_checklist"]
