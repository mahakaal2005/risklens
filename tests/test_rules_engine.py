import copy

import pytest

from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN
from ml.rules_engine import (
    ALLOWED_RECOMMENDATIONS,
    DEFAULT_RULES_PATH,
    load_rules_config,
    score_merchant_week,
)

BASE_ROW = dict(
    merchant_id="merchant_demo_0001",
    week_start="2025-06-02",
    merchant_category="apparel",
    merchant_age_days=400,
    transaction_count_30d=500,
    transaction_volume_30d=500000.0,
    transaction_volume_previous_30d=480000.0,
    refund_count_30d=8,
    top_dispute_reason_category="other",
    previous_review_outcome="none",
)

FIXTURES = {
    "stable_merchant": {
        **BASE_ROW,
        "refund_rate_30d": 0.015, "refund_rate_previous_30d": 0.014,
        "chargeback_rate_30d": 0.003, "chargeback_rate_previous_30d": 0.0028,
        "delivery_evidence_coverage": 0.92, "support_ticket_rate": 0.01,
        "average_support_resolution_time_hours": 20,
    },
    "seasonal_sale_false_positive_candidate": {
        **BASE_ROW,
        "refund_rate_30d": 0.06, "refund_rate_previous_30d": 0.02,
        "chargeback_rate_30d": 0.004, "chargeback_rate_previous_30d": 0.0035,
        "delivery_evidence_coverage": 0.88, "support_ticket_rate": 0.02,
        "average_support_resolution_time_hours": 24,
    },
    "operational_fulfilment_failure_case": {
        **BASE_ROW,
        "refund_rate_30d": 0.05, "refund_rate_previous_30d": 0.025,
        "chargeback_rate_30d": 0.012, "chargeback_rate_previous_30d": 0.009,
        "delivery_evidence_coverage": 0.60, "support_ticket_rate": 0.045,
        "average_support_resolution_time_hours": 55,
    },
    "high_risk_merchant_case": {
        **BASE_ROW,
        "refund_rate_30d": 0.07, "refund_rate_previous_30d": 0.02,
        "chargeback_rate_30d": 0.03, "chargeback_rate_previous_30d": 0.01,
        "delivery_evidence_coverage": 0.30, "support_ticket_rate": 0.07,
        "average_support_resolution_time_hours": 70,
    },
    "early_hidden_risk_case": {
        **BASE_ROW,
        "refund_rate_30d": 0.025, "refund_rate_previous_30d": 0.018,
        "chargeback_rate_30d": 0.006, "chargeback_rate_previous_30d": 0.005,
        "delivery_evidence_coverage": 0.80, "support_ticket_rate": 0.018,
        "average_support_resolution_time_hours": 30,
    },
}


def test_yaml_loads_correctly():
    config = load_rules_config()
    assert config["version"]
    rule_ids = {r["rule_id"] for r in config["rules"]}
    assert rule_ids == {
        "REFUND_RATE_SPIKE", "CHARGEBACK_RATE_SPIKE", "EVIDENCE_COVERAGE_GAP",
        "SUPPORT_OPERATIONAL_STRESS", "COMBINED_LOSS_SIGNAL",
    }


def test_stable_merchant_no_rules_trigger_low_risk():
    result = score_merchant_week(FIXTURES["stable_merchant"])
    assert result["triggered_rules"] == []
    assert result["risk_tier"] == "low"
    assert result["recommendation"] == "APPROVE"


def test_seasonal_sale_triggers_refund_spike_only_and_is_not_enforcement_like():
    result = score_merchant_week(FIXTURES["seasonal_sale_false_positive_candidate"])
    assert result["triggered_rules"] == ["REFUND_RATE_SPIKE"]
    assert result["recommendation"] in {"ALLOW_WITH_MONITORING", "REQUEST_EVIDENCE"}
    assert result["recommendation"] not in {"MANUAL_REVIEW_REQUIRED", "ESCALATE_TO_COMPLIANCE"}
    assert "CHARGEBACK_RATE_SPIKE" not in result["triggered_rules"]
    assert "COMBINED_LOSS_SIGNAL" not in result["triggered_rules"]


def test_operational_fulfilment_failure_triggers_medium_concern_and_evidence_request():
    result = score_merchant_week(FIXTURES["operational_fulfilment_failure_case"])
    assert "REFUND_RATE_SPIKE" in result["triggered_rules"]
    assert "SUPPORT_OPERATIONAL_STRESS" in result["triggered_rules"]
    assert "EVIDENCE_COVERAGE_GAP" in result["triggered_rules"]
    assert "CHARGEBACK_RATE_SPIKE" not in result["triggered_rules"]
    assert result["risk_tier"] in {"medium", "high"}
    assert result["recommendation"] == "REQUEST_EVIDENCE"


def test_high_risk_merchant_triggers_combined_signal_and_manual_review():
    result = score_merchant_week(FIXTURES["high_risk_merchant_case"])
    assert "CHARGEBACK_RATE_SPIKE" in result["triggered_rules"]
    assert "REFUND_RATE_SPIKE" in result["triggered_rules"]
    assert "COMBINED_LOSS_SIGNAL" in result["triggered_rules"]
    assert result["risk_tier"] == "high"
    assert result["recommendation"] == "MANUAL_REVIEW_REQUIRED"


def test_early_hidden_risk_case_shows_rules_are_imperfect():
    result = score_merchant_week(FIXTURES["early_hidden_risk_case"])
    assert result["risk_tier"] in {"low", "medium"}
    assert result["recommendation"] in {"APPROVE", "ALLOW_WITH_MONITORING", "REQUEST_EVIDENCE"}


@pytest.mark.parametrize("name", list(FIXTURES.keys()))
def test_score_stays_in_valid_range(name):
    result = score_merchant_week(FIXTURES[name])
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_tier"] in {"low", "medium", "high"}
    assert result["recommendation"] in ALLOWED_RECOMMENDATIONS


@pytest.mark.parametrize("name", list(FIXTURES.keys()))
def test_rule_output_has_analyst_and_merchant_safe_explanations_when_triggered(name):
    result = score_merchant_week(FIXTURES[name])
    for rule_result in result["rule_results"]:
        if rule_result.triggered:
            assert rule_result.analyst_explanation
            assert rule_result.merchant_safe_explanation
            assert rule_result.analyst_explanation != rule_result.merchant_safe_explanation


def test_input_snapshot_excludes_label_and_latent_state():
    row = {**FIXTURES["high_risk_merchant_case"], LABEL_COLUMN: 1, LATENT_STATE_COLUMN: "high_risk_merchant_behaviour"}
    result = score_merchant_week(row)
    for rule_result in result["rule_results"]:
        assert LABEL_COLUMN not in rule_result.input_snapshot
        assert LATENT_STATE_COLUMN not in rule_result.input_snapshot


def test_thresholds_are_read_from_yaml_not_hardcoded():
    config = load_rules_config()
    mutated = copy.deepcopy(config)
    for rule in mutated["rules"]:
        if rule["rule_id"] == "REFUND_RATE_SPIKE":
            rule["thresholds"]["absolute_increase_min"] = 0.99
            rule["thresholds"]["relative_multiple_min"] = 999.0

    row = FIXTURES["seasonal_sale_false_positive_candidate"]
    default_result = score_merchant_week(row, rules_config=config)
    mutated_result = score_merchant_week(row, rules_config=mutated)

    assert "REFUND_RATE_SPIKE" in default_result["triggered_rules"]
    assert "REFUND_RATE_SPIKE" not in mutated_result["triggered_rules"]


def test_missing_previous_rate_fields_fail_safely():
    row = {**FIXTURES["stable_merchant"], "chargeback_rate_previous_30d": None}
    result = score_merchant_week(row)  # should not raise
    assert "CHARGEBACK_RATE_SPIKE" not in result["triggered_rules"]


def test_engine_is_deterministic_for_same_input():
    result_a = score_merchant_week(FIXTURES["operational_fulfilment_failure_case"])
    result_b = score_merchant_week(FIXTURES["operational_fulfilment_failure_case"])
    assert result_a["risk_score"] == result_b["risk_score"]
    assert result_a["risk_tier"] == result_b["risk_tier"]
    assert result_a["recommendation"] == result_b["recommendation"]
    assert result_a["triggered_rules"] == result_b["triggered_rules"]


def test_default_rules_path_exists():
    assert DEFAULT_RULES_PATH.exists()
