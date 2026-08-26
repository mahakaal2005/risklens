import numpy as np
import pytest

from ml.model_utils import (
    ALLOWED_RECOMMENDATIONS,
    check_near_perfect,
    combined_policy,
    compute_metrics,
    rules_only_positive_prediction,
)
from ml.rules_engine import score_merchant_week

BASE_ROW = dict(
    merchant_id="merchant_demo_0001", week_start="2025-06-02", merchant_category="apparel",
    merchant_age_days=400, transaction_count_30d=500, transaction_volume_30d=500000.0,
    transaction_volume_previous_30d=480000.0, refund_count_30d=8,
    top_dispute_reason_category="other", previous_review_outcome="none",
)

STABLE_FIXTURE = {**BASE_ROW, "refund_rate_30d": 0.015, "refund_rate_previous_30d": 0.014,
                  "chargeback_rate_30d": 0.003, "chargeback_rate_previous_30d": 0.0028,
                  "delivery_evidence_coverage": 0.92, "support_ticket_rate": 0.01,
                  "average_support_resolution_time_hours": 20}

HIGH_RISK_FIXTURE = {**BASE_ROW, "refund_rate_30d": 0.07, "refund_rate_previous_30d": 0.02,
                     "chargeback_rate_30d": 0.03, "chargeback_rate_previous_30d": 0.01,
                     "delivery_evidence_coverage": 0.30, "support_ticket_rate": 0.07,
                     "average_support_resolution_time_hours": 70}


def test_compute_metrics_returns_valid_values_and_confusion_matrix_shape():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.6, 0.2, 0.7, 0.3, 0.9, 0.8, 0.05])

    metrics = compute_metrics(y_true, y_pred, y_score)

    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["f2"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["roc_auc_secondary"] <= 1
    assert 0 <= metrics["false_positive_rate"] <= 1
    assert 0 <= metrics["false_negative_rate"] <= 1
    cm = metrics["confusion_matrix"]
    assert set(cm.keys()) == {"tn", "fp", "fn", "tp"}
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_true)


def test_compute_metrics_without_score_leaves_auc_fields_none():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = compute_metrics(y_true, y_pred, y_score=None)
    assert metrics["pr_auc"] is None
    assert metrics["roc_auc_secondary"] is None


def test_rules_only_baseline_stable_case_is_negative():
    result = score_merchant_week(STABLE_FIXTURE)
    assert rules_only_positive_prediction(result) is False


def test_rules_only_baseline_high_risk_case_is_positive():
    result = score_merchant_week(HIGH_RISK_FIXTURE)
    assert rules_only_positive_prediction(result) is True


@pytest.mark.parametrize("ml_probability", [0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0])
@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("triggered_rules", [
    set(),
    {"REFUND_RATE_SPIKE"},
    {"CHARGEBACK_RATE_SPIKE"},
    {"CHARGEBACK_RATE_SPIKE", "COMBINED_LOSS_SIGNAL"},
    {"EVIDENCE_COVERAGE_GAP", "SUPPORT_OPERATIONAL_STRESS"},
])
def test_combined_policy_output_is_always_an_allowed_recommendation(ml_probability, threshold, triggered_rules):
    decision = combined_policy(ml_probability, threshold, triggered_rules)
    assert decision.recommendation in ALLOWED_RECOMMENDATIONS
    assert decision.risk_signal_intensity in {"Low", "Medium", "High"}
    assert decision.policy_explanation


def test_near_perfect_gate_flags_zero_false_positives():
    metrics = {
        "pr_auc": 0.5, "precision": 1.0, "recall": 0.4,
        "confusion_matrix": {"tn": 100, "fp": 0, "fn": 6, "tp": 4},
    }
    triggered, reason = check_near_perfect(metrics)
    assert triggered is True
    assert "false positives" in reason or "fp=0" in reason


def test_near_perfect_gate_flags_high_precision_and_recall():
    metrics = {
        "pr_auc": 0.5, "precision": 0.99, "recall": 0.99,
        "confusion_matrix": {"tn": 100, "fp": 1, "fn": 1, "tp": 98},
    }
    triggered, reason = check_near_perfect(metrics)
    assert triggered is True


def test_near_perfect_gate_flags_high_pr_auc():
    metrics = {
        "pr_auc": 0.99, "precision": 0.6, "recall": 0.6,
        "confusion_matrix": {"tn": 100, "fp": 20, "fn": 20, "tp": 30},
    }
    triggered, reason = check_near_perfect(metrics)
    assert triggered is True


def test_near_perfect_gate_does_not_flag_normal_result():
    metrics = {
        "pr_auc": 0.65, "precision": 0.53, "recall": 0.86,
        "confusion_matrix": {"tn": 1360, "fp": 174, "fn": 31, "tp": 195},
    }
    triggered, reason = check_near_perfect(metrics)
    assert triggered is False
