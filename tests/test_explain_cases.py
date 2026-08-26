import joblib
import pandas as pd
import pytest

from ml.explain_cases import build_analyst_summary, build_trend_values, build_uncertainty_statement, compute_top_factors
from ml.features import compute_features
from ml.model_utils import ML_FEATURE_COLUMNS

ARTIFACT_PATH = "ml/artifacts/logistic_regression_v0.1.0.joblib"

BASE_ROW = dict(
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


def _row_df():
    features = compute_features(BASE_ROW)
    return pd.DataFrame([features])[ML_FEATURE_COLUMNS]


def test_compute_top_factors_returns_ranked_list(pipeline):
    factors = compute_top_factors(pipeline, _row_df())
    assert len(factors) > 0
    magnitudes = [abs(f["contribution"]) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_top_factors_only_reference_known_features(pipeline):
    factors = compute_top_factors(pipeline, _row_df())
    known_prefixes = tuple(ML_FEATURE_COLUMNS)
    for f in factors:
        assert f["feature"].startswith(known_prefixes) or f["feature"] in ML_FEATURE_COLUMNS or "_" in f["feature"]


def test_top_factor_sentences_use_directional_not_causal_language(pipeline):
    factors = compute_top_factors(pipeline, _row_df())
    forbidden = ["caused fraud", "proves", "confirmed abuse", "guarantees"]
    for f in factors:
        sentence_lower = f["sentence"].lower()
        for term in forbidden:
            assert term not in sentence_lower
        assert "contributed to" in sentence_lower


def test_compute_top_factors_is_deterministic(pipeline):
    factors_a = compute_top_factors(pipeline, _row_df())
    factors_b = compute_top_factors(pipeline, _row_df())
    assert factors_a == factors_b


def test_build_uncertainty_statement_degraded_mode():
    statement = build_uncertainty_statement(None, degraded_mode=True)
    assert "rules engine only" in statement or "degraded" in statement.lower()


def test_build_uncertainty_statement_normal_mode():
    statement = build_uncertainty_statement(0.5, degraded_mode=False)
    assert "not a confirmed finding of fraud" in statement or "not guarantee" in statement.lower()


def test_build_trend_values_contains_required_fields():
    features = compute_features(BASE_ROW)
    trends = build_trend_values(BASE_ROW, features)
    assert "refund_rate_current_vs_prior" in trends
    assert "chargeback_rate_current_vs_prior" in trends
    assert "transaction_volume_current_vs_prior" in trends
    assert "delivery_evidence_coverage" in trends
    assert "support_ticket_rate" in trends
    assert "support_resolution_time_hours" in trends


def test_build_analyst_summary_is_at_most_four_sentences(pipeline):
    factors = compute_top_factors(pipeline, _row_df())
    summary = build_analyst_summary(["REFUND_RATE_SPIKE", "CHARGEBACK_RATE_SPIKE"], factors, "MANUAL_REVIEW_REQUIRED", degraded_mode=False)
    sentence_count = summary.count(". ") + (1 if summary.strip().endswith(".") else 0)
    assert sentence_count <= 4


def test_build_analyst_summary_degraded_mode_has_no_model_factor_sentence():
    summary = build_analyst_summary(["REFUND_RATE_SPIKE"], [], "REQUEST_EVIDENCE", degraded_mode=True)
    assert "Recommended action: REQUEST_EVIDENCE." in summary
