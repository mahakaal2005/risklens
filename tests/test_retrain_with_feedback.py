"""Tests for ml/retrain_with_feedback.py -- the manually-triggered
feedback retraining loop. See docs/PHASE_2_FEEDBACK_RETRAINING_DESIGN.md."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ml.generate_synthetic_data import generate_dataset
from ml.retrain_with_feedback import apply_feedback_to_training_split, load_feedback_overrides, retrain_with_feedback
from ml.split_data import load_and_split
from ml.train_baseline_model import train


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    df = generate_dataset(seed=42, n_merchants=60, n_weeks=52)
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def baseline_artifacts(tmp_path_factory, synthetic_csv):
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    train(csv_path=synthetic_csv, seed=42, artifact_dir=artifact_dir)
    return artifact_dir


def test_load_feedback_overrides_returns_empty_list_for_missing_file(tmp_path):
    assert load_feedback_overrides(tmp_path / "does_not_exist.json") == []


def test_apply_feedback_overrides_a_matching_training_row(synthetic_csv):
    split = load_and_split(synthetic_csv)
    train_df = split["train"]
    row = train_df.iloc[0]
    original_label = int(row["label_high_loss_next_30d"])
    corrected_label = 1 - original_label

    overrides = [{
        "case_id": "c1", "merchant_id": row["merchant_id"], "week_start": row["week_start"],
        "final_outcome": "FALSE_POSITIVE" if corrected_label == 0 else "CONFIRMED_RISK",
        "corrected_label": corrected_label,
    }]
    corrected_df, report = apply_feedback_to_training_split(train_df, overrides)

    assert len(report["applied"]) == 1
    assert len(report["skipped"]) == 0
    mask = (corrected_df["merchant_id"] == row["merchant_id"]) & (corrected_df["week_start"] == row["week_start"])
    assert int(corrected_df.loc[mask, "label_high_loss_next_30d"].iloc[0]) == corrected_label


def test_apply_feedback_skips_override_for_a_week_outside_training_split(synthetic_csv):
    split = load_and_split(synthetic_csv)
    train_df, test_df = split["train"], split["test"]
    test_row = test_df.iloc[0]

    overrides = [{
        "case_id": "c1", "merchant_id": test_row["merchant_id"], "week_start": test_row["week_start"],
        "final_outcome": "FALSE_POSITIVE", "corrected_label": 0,
    }]
    corrected_df, report = apply_feedback_to_training_split(train_df, overrides)

    assert len(report["applied"]) == 0
    assert len(report["skipped"]) == 1
    assert "No matching row" in report["skipped"][0]["reason"]
    pd.testing.assert_frame_equal(corrected_df, train_df)


def test_apply_feedback_skips_when_label_already_matches(synthetic_csv):
    split = load_and_split(synthetic_csv)
    train_df = split["train"]
    row = train_df.iloc[0]
    original_label = int(row["label_high_loss_next_30d"])

    overrides = [{
        "case_id": "c1", "merchant_id": row["merchant_id"], "week_start": row["week_start"],
        "final_outcome": "CONFIRMED_RISK" if original_label == 1 else "FALSE_POSITIVE",
        "corrected_label": original_label,
    }]
    _corrected_df, report = apply_feedback_to_training_split(train_df, overrides)
    assert len(report["applied"]) == 0
    assert len(report["skipped"]) == 1
    assert "already matches" in report["skipped"][0]["reason"]


def test_retrain_with_no_feedback_file_still_produces_a_report(synthetic_csv, baseline_artifacts, tmp_path):
    missing_feedback_path = tmp_path / "no_feedback.json"
    report = retrain_with_feedback(
        csv_path=synthetic_csv, feedback_path=missing_feedback_path, seed=42, artifact_dir=baseline_artifacts,
    )
    assert report["feedback_summary"]["total_overrides_available"] == 0
    assert report["feedback_summary"]["applied_to_training"] == 0
    assert "before_feedback" in report["held_out_test_comparison"]
    assert "after_feedback" in report["held_out_test_comparison"]


def test_retrain_raises_if_no_baseline_model_exists(synthetic_csv, tmp_path):
    empty_artifact_dir = tmp_path / "empty_artifacts"
    empty_artifact_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No baseline model found"):
        retrain_with_feedback(csv_path=synthetic_csv, artifact_dir=empty_artifact_dir)


def test_retrain_applies_feedback_and_saves_a_distinct_model_version(synthetic_csv, baseline_artifacts, tmp_path):
    split = load_and_split(synthetic_csv)
    train_df = split["train"]
    row = train_df.iloc[0]
    original_label = int(row["label_high_loss_next_30d"])
    corrected_label = 1 - original_label

    feedback_path = tmp_path / "feedback.json"
    with open(feedback_path, "w") as f:
        json.dump([{
            "case_id": "c1", "merchant_id": row["merchant_id"], "week_start": row["week_start"],
            "final_outcome": "FALSE_POSITIVE" if corrected_label == 0 else "CONFIRMED_RISK",
            "corrected_label": corrected_label,
        }], f)

    report = retrain_with_feedback(
        csv_path=synthetic_csv, feedback_path=feedback_path, seed=42, artifact_dir=baseline_artifacts,
    )

    assert report["feedback_summary"]["applied_to_training"] == 1
    assert (baseline_artifacts / "logistic_regression_v0.1.0-feedback1.joblib").exists()
    assert (baseline_artifacts / "feedback_retrain_report.json").exists()
    # The baseline model file must be untouched -- retraining must never
    # overwrite it.
    assert (baseline_artifacts / "logistic_regression_v0.1.0.joblib").exists()


def test_report_never_claims_the_test_split_was_touched(synthetic_csv, baseline_artifacts, tmp_path):
    report = retrain_with_feedback(
        csv_path=synthetic_csv, feedback_path=tmp_path / "no_feedback.json", seed=42, artifact_dir=baseline_artifacts,
    )
    assert "untouched" in report["held_out_test_comparison"]["note"]
    assert "manually triggered" in report["synthetic_data_notice"]
