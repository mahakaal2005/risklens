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

# Same small-scale fixture sizing as tests/test_trajectory_transformer.py --
# large enough for a real chronological split, small enough that a test can
# run the full evaluate() pipeline without the 93,600-row production CSV.
SMALL_N_MERCHANTS = 40
SMALL_N_WEEKS = 20

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


class _ConstantPipeline:
    """Minimal stand-in for a fitted sklearn pipeline, so the precomputed_probs
    fallback can be tested without training a real model."""

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, X):
        column = np.full(len(X), self.probability)
        return np.column_stack([1.0 - column, column])


@pytest.fixture(scope="module")
def evaluation_frame():
    from ml.generate_synthetic_data import generate_dataset

    return generate_dataset(seed=42, n_merchants=SMALL_N_MERCHANTS, n_weeks=SMALL_N_WEEKS)


def test_evaluate_split_uses_precomputed_probs_instead_of_a_pipeline(evaluation_frame):
    """A method supplied via precomputed_probs is scored from those
    probabilities, not from any pipeline -- this is the path the Trajectory
    Transformer uses, since it scores whole merchant sequences and cannot
    expose .predict_proba(X) on a design matrix."""
    from ml.evaluate_model import evaluate_split

    # Everything above the threshold: the method must flag every row.
    probabilities = np.full(len(evaluation_frame), 0.99)

    results, _ = evaluate_split(
        evaluation_frame,
        {"ml_only": _ConstantPipeline(0.01)},
        threshold=0.5,
        precomputed_probs={"trajectory_transformer": probabilities},
    )

    assert "trajectory_transformer" in results
    assert results["trajectory_transformer"]["n_predicted_positive"] == len(evaluation_frame)
    # The sklearn-backed method is unaffected and still uses its own pipeline.
    assert results["ml_only"]["n_predicted_positive"] == 0


def test_evaluate_split_falls_back_to_the_sklearn_path_without_precomputed_probs(evaluation_frame):
    from ml.evaluate_model import evaluate_split

    results, _ = evaluate_split(evaluation_frame, {"ml_only": _ConstantPipeline(0.99)}, threshold=0.5)

    assert "trajectory_transformer" not in results
    assert results["ml_only"]["n_predicted_positive"] == len(evaluation_frame)


def test_evaluate_split_scores_each_method_at_its_own_threshold(evaluation_frame):
    """Regression test for the threshold-scoring bug: applying one model's
    operating point to another model's probabilities is not a like-for-like
    comparison. 0.6 must clear its own 0.5 threshold while failing the shared
    0.9 fallback."""
    from ml.evaluate_model import evaluate_split

    results, _ = evaluate_split(
        evaluation_frame,
        {"ml_only": _ConstantPipeline(0.95), "random_forest": _ConstantPipeline(0.6)},
        threshold=0.9,
        thresholds_by_method={"ml_only": 0.9, "random_forest": 0.5},
    )

    assert results["random_forest"]["threshold_used"] == 0.5
    assert results["random_forest"]["n_predicted_positive"] == len(evaluation_frame)
    assert results["ml_only"]["threshold_used"] == 0.9
    # Rules fire on their own conditions, not a probability cut point.
    assert results["rules_only"]["threshold_used"] is None


def test_evaluate_omits_trajectory_transformer_when_the_artifact_is_absent(tmp_path, capsys):
    """A missing comparison-model artifact must degrade to a clear message and
    a report without that method -- never an exception. Same graceful-skip
    contract the tree models already have.

    Runs the real evaluate() end-to-end (rules scoring, sklearn scoring,
    combined policy, scenario difficulty, near-perfect gate -- nothing stubbed)
    but over the small fixture dataset, since this asserts a control-flow
    branch rather than any metric value. On the full 93,600-row CSV this single
    test costs minutes.
    """
    import shutil
    from pathlib import Path

    from ml.evaluate_model import DEFAULT_ARTIFACT_DIR, evaluate
    from ml.generate_synthetic_data import generate_dataset

    csv_path = tmp_path / "small_dataset.csv"
    generate_dataset(seed=42, n_merchants=SMALL_N_MERCHANTS, n_weeks=SMALL_N_WEEKS).to_csv(csv_path, index=False)

    # Copy only the artifacts the required path needs, deliberately omitting
    # every optional comparison model.
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    for name in ("logistic_regression_v0.1.0.joblib", "logistic_regression_v0.1.0_metadata.json"):
        shutil.copy(Path(DEFAULT_ARTIFACT_DIR) / name, artifact_dir / name)

    result = evaluate(csv_path=csv_path, artifact_dir=artifact_dir)

    assert "trajectory_transformer" not in result["test_results"]
    assert "trajectory_transformer" not in result["validation_results"]
    assert "ml_only" in result["test_results"]  # the required path still ran
    assert result["test_results"]["ml_only"]["confusion_matrix"]  # scored real rows, not a stub
    assert "Trajectory Transformer artifact not found" in capsys.readouterr().out


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
