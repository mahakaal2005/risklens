from pathlib import Path

import pandas as pd
import pytest

from ml.data_validation import DataValidationError
from ml.generate_synthetic_data import generate_dataset
from ml.inspect_synthetic_data import (
    CANDIDATE_MODEL_FEATURES,
    LABEL_COLUMN,
    LATENT_STATE_COLUMN,
    generate_report,
    scan_leakage_column_names,
)


def test_latent_state_excluded_from_candidate_features():
    assert LATENT_STATE_COLUMN not in CANDIDATE_MODEL_FEATURES


def test_label_excluded_from_candidate_features():
    assert LABEL_COLUMN not in CANDIDATE_MODEL_FEATURES


def test_report_creates_successfully(tmp_path):
    df = generate_dataset(seed=42, n_merchants=220, n_weeks=52)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    report_path = tmp_path / "report.md"

    result = generate_report(csv_path=csv_path, report_path=report_path)

    assert report_path.exists()
    content = report_path.read_text()
    assert "Milestone 1 Data Quality Report" in content
    assert result["decision"] in {"APPROVED FOR MILESTONE 2", "UNDER INVESTIGATION"}


def test_leakage_scan_flags_injected_forbidden_column():
    columns = ["merchant_id", "week_start", "refund_rate_30d", LABEL_COLUMN, LATENT_STATE_COLUMN, "future_chargeback_next_30d"]
    result = scan_leakage_column_names(columns)
    assert "future" in result["unexplained_matches"]
    assert "future_chargeback_next_30d" in result["unexplained_matches"]["future"]


def test_leakage_scan_does_not_flag_allowed_exceptions():
    columns = ["merchant_id", LABEL_COLUMN, LATENT_STATE_COLUMN, "previous_review_outcome"]
    result = scan_leakage_column_names(columns)
    assert result["unexplained_matches"] == {}
    assert "previous_review_outcome" in result["reviewed_not_leakage"]


def test_leakage_scan_flags_risk_state_like_column():
    columns = ["merchant_id", "hidden_risk_state_score"]
    result = scan_leakage_column_names(columns)
    assert "risk_state" in result["unexplained_matches"]
    assert "hidden_risk_state_score" in result["unexplained_matches"]["risk_state"]
