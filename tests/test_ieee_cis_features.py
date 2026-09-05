"""Tests for external_benchmark.ieee_cis_features, ieee_cis_train's
chronological split, and ieee_cis_evaluate's aggregate-only report shape.
Never requires the real Kaggle dataset -- every fixture is a tiny
hand-built DataFrame."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from external_benchmark import ieee_cis_evaluate, ieee_cis_features, ieee_cis_train


def _valid_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1000, 1000 + n),
            "TransactionDT": np.sort(rng.integers(0, 30 * 86400, size=n)),
            "TransactionAmt": rng.uniform(10, 500, size=n).round(2),
            "isFraud": rng.integers(0, 2, size=n),
            "ProductCD": rng.choice(["W", "C", "R", "H"], size=n),
        }
    )


def test_feature_frame_has_required_numeric_features():
    df = _valid_df()
    features = ieee_cis_features.build_feature_frame(df)
    for col in ieee_cis_features.REQUIRED_NUMERIC_FEATURES:
        assert col in features.columns


def test_transaction_amt_has_cents_detected_correctly():
    df = pd.DataFrame({"TransactionDT": [0, 86400], "TransactionAmt": [100.0, 100.50], "isFraud": [0, 0]})
    features = ieee_cis_features.build_feature_frame(df)
    assert features["transaction_amt_has_cents"].tolist() == [0, 1]


def test_target_excluded_from_feature_matrix():
    df = _valid_df()
    features = ieee_cis_features.build_feature_frame(df)
    assert "isFraud" not in features.columns


def test_feature_selection_excludes_identity_and_id_like_columns():
    df = _valid_df()
    features = ieee_cis_features.build_feature_frame(df)
    numeric, categorical = ieee_cis_features.selected_feature_columns(features)
    all_selected = numeric + categorical
    assert "TransactionID" not in all_selected
    assert not any(c.startswith("id_") for c in all_selected)
    assert "isFraud" not in all_selected


def test_save_feature_list_writes_expected_json(tmp_path):
    df = _valid_df()
    features = ieee_cis_features.build_feature_frame(df)
    out_path = tmp_path / "selected_feature_list.json"
    payload = ieee_cis_features.save_feature_list(features, out_path)

    assert out_path.exists()
    with out_path.open() as f:
        saved = json.load(f)
    assert saved == payload
    assert "numeric_features" in saved
    assert "categorical_features" in saved


def test_chronological_split_has_no_time_overlap():
    df = _valid_df(n=100)
    train_df, val_df, test_df = ieee_cis_train.chronological_split(df)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert train_df["TransactionDT"].max() <= val_df["TransactionDT"].min()
    assert val_df["TransactionDT"].max() <= test_df["TransactionDT"].min()


def test_chronological_split_uses_no_shuffling():
    df = _valid_df(n=50)
    train_df, _, _ = ieee_cis_train.chronological_split(df)
    assert train_df["TransactionDT"].is_monotonic_increasing


def test_select_threshold_never_uses_test_data():
    y_val = pd.Series([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    probs_val = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.4, 0.7, 0.05, 0.15, 0.6])
    result = ieee_cis_train.select_threshold(y_val, probs_val)
    assert "grid" in result
    assert "selected" in result
    assert 0.0 <= result["selected"]["threshold"] <= 1.0


def test_select_threshold_fallback_when_no_signal():
    y_val = pd.Series([0, 0, 0, 0, 0])
    probs_val = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = ieee_cis_train.select_threshold(y_val, probs_val)
    assert result["selected"]["fallback_used"] is True


def test_full_benchmark_report_contains_only_aggregate_metrics(tmp_path, monkeypatch):
    df = _valid_df(n=200)
    dataset_path = tmp_path / "train_transaction.csv"
    df.to_csv(dataset_path, index=False)

    artifact_path = tmp_path / "latest_ieee_cis_report.json"
    feature_list_path = tmp_path / "selected_feature_list.json"
    monkeypatch.setattr(ieee_cis_evaluate, "ARTIFACT_PATH", artifact_path)
    monkeypatch.setattr(ieee_cis_evaluate, "FEATURE_LIST_PATH", feature_list_path)

    report = ieee_cis_evaluate.run_full_benchmark(dataset_path)

    assert artifact_path.exists()
    with artifact_path.open() as f:
        saved_report = json.load(f)
    assert saved_report == report

    serialized = json.dumps(report)
    assert "TransactionID" not in serialized
    assert "1000" not in serialized  # smallest synthetic TransactionID value
    for expected_key in ["fixture_label", "non_claims", "split", "selected_features", "threshold_selection", "test_metrics", "test_class_prevalence"]:
        assert expected_key in report


def test_report_contains_required_non_claim_text(tmp_path, monkeypatch):
    df = _valid_df(n=200)
    dataset_path = tmp_path / "train_transaction.csv"
    df.to_csv(dataset_path, index=False)

    monkeypatch.setattr(ieee_cis_evaluate, "ARTIFACT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(ieee_cis_evaluate, "FEATURE_LIST_PATH", tmp_path / "features.json")

    report = ieee_cis_evaluate.run_full_benchmark(dataset_path)
    non_claims = report["non_claims"]
    assert "does not validate RiskLens" in non_claims
    assert "not Razorpay data" in non_claims
    assert "not proof of production fraud performance" in non_claims
