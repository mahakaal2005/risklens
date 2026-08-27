"""Tests for ml/external_data_import.py -- validating and scoring an
anonymized merchant-week CSV import. See
docs/PHASE_2_EXTERNAL_DATA_IMPORT_DESIGN.md."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.external_data_import import (
    EXTERNAL_IMPORT_DATA_NOTICE,
    ExternalImportValidationError,
    IMPORT_REQUIRED_COLUMNS,
    build_mapping_report,
    score_import_rows,
    validate_import_dataframe,
)
from ml.rules_engine import DEFAULT_RULES_PATH, load_rules_config


def _valid_row(**overrides) -> dict:
    row = {
        "merchant_id": "ext_merchant_0001", "week_start": "2025-06-02",
        "merchant_category": "electronics", "merchant_age_days": 900,
        "transaction_count_30d": 420, "transaction_volume_30d": 84000.0,
        "transaction_volume_previous_30d": 82000.0, "transaction_volume_change_30d": 2000.0,
        "refund_count_30d": 8, "refund_rate_30d": 0.019, "refund_rate_previous_30d": 0.018,
        "refund_rate_change_30d": 0.001,
        "chargeback_count_30d": 1, "chargeback_rate_30d": 0.0024, "chargeback_rate_previous_30d": 0.0022,
        "chargeback_rate_change_30d": 0.0002,
        "top_dispute_reason_category": "other", "delivery_evidence_coverage": 0.95,
        "support_ticket_rate": 0.01, "average_support_resolution_time_hours": 6.0,
        "previous_review_outcome": "none",
    }
    row.update(overrides)
    return row


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_valid_dataframe_passes_validation():
    validate_import_dataframe(_df([_valid_row()]))  # no raise


def test_missing_required_column_is_rejected():
    row = _valid_row()
    del row["refund_rate_30d"]
    with pytest.raises(ExternalImportValidationError, match="Missing required columns"):
        validate_import_dataframe(_df([row]))


def test_exact_prohibited_field_name_is_rejected():
    row = _valid_row()
    row["email"] = "someone@example.com"
    with pytest.raises(ExternalImportValidationError, match="Prohibited/PII-suggestive"):
        validate_import_dataframe(_df([row]))


def test_pii_suspicious_substring_column_is_rejected():
    """The exact-match prohibited-field list alone wouldn't catch this --
    this is the substring check's job."""
    row = _valid_row()
    row["customer_email"] = "someone@example.com"
    with pytest.raises(ExternalImportValidationError, match="rejected on suspicion"):
        validate_import_dataframe(_df([row]))


def test_label_column_present_is_rejected():
    row = _valid_row()
    row["label_high_loss_next_30d"] = 1
    with pytest.raises(ExternalImportValidationError, match="own simulation"):
        validate_import_dataframe(_df([row]))


def test_latent_state_column_present_is_rejected():
    row = _valid_row()
    row["latent_state_for_demo_only"] = "stable_merchant"
    with pytest.raises(ExternalImportValidationError, match="own simulation"):
        validate_import_dataframe(_df([row]))


def test_unsafe_merchant_id_is_rejected():
    row = _valid_row(merchant_id="../../etc/passwd")
    with pytest.raises(ExternalImportValidationError, match="safe anonymized tokens"):
        validate_import_dataframe(_df([row]))


def test_bad_week_start_format_is_rejected():
    row = _valid_row(week_start="not-a-date")
    with pytest.raises(ExternalImportValidationError, match="do not parse as a"):
        validate_import_dataframe(_df([row]))


def test_negative_value_in_non_negative_column_is_rejected():
    row = _valid_row(refund_count_30d=-1)
    with pytest.raises(ExternalImportValidationError, match="negative"):
        validate_import_dataframe(_df([row]))


def test_rate_column_outside_0_1_is_rejected():
    row = _valid_row(refund_rate_30d=1.5)
    with pytest.raises(ExternalImportValidationError, match="0.0-1.0"):
        validate_import_dataframe(_df([row]))


def test_duplicate_merchant_week_is_rejected():
    row = _valid_row()
    with pytest.raises(ExternalImportValidationError, match="duplicate"):
        validate_import_dataframe(_df([row, dict(row)]))


def test_mapping_report_lists_found_and_missing_columns():
    row = _valid_row()
    del row["refund_rate_30d"]
    report = build_mapping_report(_df([row]))
    assert "refund_rate_30d" in report["required_columns_missing"]
    assert "merchant_id" in report["required_columns_found"]
    assert report["row_count"] == 1


def test_mapping_report_never_includes_a_raw_row():
    report = build_mapping_report(_df([_valid_row()]))
    serialized = str(report)
    assert "ext_merchant_0001" not in serialized  # no raw merchant_id/row value, aggregate-only


def test_score_import_rows_produces_a_packet_with_import_notice_not_synthetic_notice():
    rules_config = load_rules_config(DEFAULT_RULES_PATH)
    df = _df([_valid_row()])
    packets = score_import_rows(df, pipeline=None, rules_config=rules_config, threshold=0.1, model_version=None, rules_version=rules_config["version"])
    assert len(packets) == 1
    notice = packets[0]["identification"]["synthetic_data_notice"]
    assert notice == EXTERNAL_IMPORT_DATA_NOTICE
    assert "synthetic, demonstration-only data" not in notice
    assert packets[0]["identification"]["data_source"] == "external_csv_import"


def test_score_import_rows_never_leaks_label_or_latent_state():
    rules_config = load_rules_config(DEFAULT_RULES_PATH)
    df = _df([_valid_row()])
    packets = score_import_rows(df, pipeline=None, rules_config=rules_config, threshold=0.1, model_version=None, rules_version=rules_config["version"])
    serialized = str(packets)
    assert "label_high_loss_next_30d" not in serialized
    assert "latent_state_for_demo_only" not in serialized


def test_score_import_rows_degraded_mode_without_a_model():
    rules_config = load_rules_config(DEFAULT_RULES_PATH)
    df = _df([_valid_row()])
    packets = score_import_rows(df, pipeline=None, rules_config=rules_config, threshold=0.1, model_version=None, rules_version=rules_config["version"])
    assert packets[0]["assessment"]["degraded_mode"] is True
    assert packets[0]["assessment"]["model_probability"] is None


def test_required_columns_exclude_label_and_latent_state():
    assert "label_high_loss_next_30d" not in IMPORT_REQUIRED_COLUMNS
    assert "latent_state_for_demo_only" not in IMPORT_REQUIRED_COLUMNS
