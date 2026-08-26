import pandas as pd
import pytest

from ml.features import (
    EXCLUDED_INPUT_COLUMNS,
    FEATURE_CATALOGUE,
    LABEL_COLUMN,
    LATENT_STATE_COLUMN,
    compute_feature_frame,
    compute_features,
)

BASE_ROW = {
    "merchant_id": "merchant_demo_0001",
    "week_start": "2025-06-02",
    "merchant_category": "apparel",
    "merchant_age_days": 400,
    "transaction_count_30d": 500,
    "transaction_volume_30d": 500000.0,
    "transaction_volume_previous_30d": 480000.0,
    "refund_count_30d": 8,
    "refund_rate_30d": 0.02,
    "refund_rate_previous_30d": 0.015,
    "chargeback_count_30d": 3,
    "chargeback_rate_30d": 0.006,
    "chargeback_rate_previous_30d": 0.004,
    "top_dispute_reason_category": "other",
    "delivery_evidence_coverage": 0.85,
    "support_ticket_rate": 0.02,
    "average_support_resolution_time_hours": 30,
    "previous_review_outcome": "none",
    LABEL_COLUMN: 1,
    LATENT_STATE_COLUMN: "stable_merchant",
}


def test_refund_rate_change_calculation():
    features = compute_features(BASE_ROW)
    assert features["refund_rate_change"] == pytest.approx(0.005, abs=1e-6)


def test_chargeback_rate_change_calculation():
    features = compute_features(BASE_ROW)
    assert features["chargeback_rate_change"] == pytest.approx(0.002, abs=1e-6)


def test_transaction_volume_change_calculation():
    features = compute_features(BASE_ROW)
    expected = (500000.0 - 480000.0) / 480000.0
    assert features["transaction_volume_change"] == pytest.approx(expected, abs=1e-4)


def test_transaction_volume_change_handles_zero_previous_volume():
    row = {**BASE_ROW, "transaction_volume_previous_30d": 0.0}
    features = compute_features(row)
    assert features["transaction_volume_change"] == 0.0


def test_delivery_evidence_gap_calculation():
    features = compute_features(BASE_ROW)
    assert features["delivery_evidence_gap"] == pytest.approx(0.15, abs=1e-6)


def test_refund_to_chargeback_ratio_computed_when_safe():
    features = compute_features(BASE_ROW)
    expected = round(0.02 / 0.006, 4)
    assert features["refund_to_chargeback_ratio"] == pytest.approx(expected, abs=1e-3)


def test_refund_to_chargeback_ratio_is_missing_when_chargeback_rate_too_small():
    row = {**BASE_ROW, "chargeback_rate_30d": 0.0005}
    features = compute_features(row)
    assert features["refund_to_chargeback_ratio"] is None


def test_refund_to_chargeback_ratio_is_capped():
    row = {**BASE_ROW, "refund_rate_30d": 0.5, "chargeback_rate_30d": 0.002}
    features = compute_features(row)
    assert features["refund_to_chargeback_ratio"] <= 50.0


def test_support_resolution_band_boundaries():
    assert compute_features({**BASE_ROW, "average_support_resolution_time_hours": 10})["support_resolution_risk_band"] == "low"
    assert compute_features({**BASE_ROW, "average_support_resolution_time_hours": 30})["support_resolution_risk_band"] == "medium"
    assert compute_features({**BASE_ROW, "average_support_resolution_time_hours": 60})["support_resolution_risk_band"] == "high"


def test_merchant_age_band_boundaries():
    assert compute_features({**BASE_ROW, "merchant_age_days": 20})["merchant_age_band"] == "new"
    assert compute_features({**BASE_ROW, "merchant_age_days": 200})["merchant_age_band"] == "growing"
    assert compute_features({**BASE_ROW, "merchant_age_days": 1000})["merchant_age_band"] == "established"


def test_no_target_or_latent_state_in_feature_output():
    features = compute_features(BASE_ROW)
    assert LABEL_COLUMN not in features
    assert LATENT_STATE_COLUMN not in features


def test_feature_output_excludes_target_even_if_input_has_extreme_values():
    row = {**BASE_ROW, LABEL_COLUMN: 1, LATENT_STATE_COLUMN: "high_risk_merchant_behaviour"}
    features = compute_features(row)
    assert set(features.keys()).isdisjoint(EXCLUDED_INPUT_COLUMNS)


def test_feature_catalogue_entries_have_required_metadata():
    required_keys = {"name", "formula", "meaning", "range", "missing_value_behavior", "used_by"}
    for entry in FEATURE_CATALOGUE:
        assert required_keys.issubset(entry.keys())
        assert entry["used_by"] in {"rules", "ml", "both"}


def test_compute_feature_frame_matches_row_level_computation():
    df = pd.DataFrame([BASE_ROW, BASE_ROW])
    frame = compute_feature_frame(df)
    row_features = compute_features(BASE_ROW)
    assert frame.loc[0, "refund_rate_change"] == pytest.approx(row_features["refund_rate_change"], abs=1e-4)
    assert frame.loc[0, "delivery_evidence_gap"] == pytest.approx(row_features["delivery_evidence_gap"], abs=1e-4)


def test_compute_feature_frame_excludes_target_and_latent_state():
    df = pd.DataFrame([BASE_ROW])
    frame = compute_feature_frame(df)
    assert LABEL_COLUMN not in frame.columns
    assert LATENT_STATE_COLUMN not in frame.columns
