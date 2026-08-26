"""Tests for external_benchmark.ieee_cis_validate and the loader's missing
-file behavior. Never requires the real Kaggle dataset -- every fixture is
a tiny hand-built DataFrame or CSV written to a temporary directory."""

from __future__ import annotations

import pandas as pd
import pytest

from external_benchmark import ieee_cis_loader, ieee_cis_validate


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionDT": [86400, 90000, 172800, 176400],
            "TransactionAmt": [100.0, 50.5, 75.25, 20.0],
            "isFraud": [0, 1, 0, 0],
            "ProductCD": ["W", "C", "W", "R"],
        }
    )


def test_missing_dataset_produces_clear_manual_download_message(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(ieee_cis_loader.DatasetNotFoundError) as exc_info:
        ieee_cis_loader.load_train_transaction(missing_path)

    message = str(exc_info.value)
    assert str(missing_path) in message
    assert "manually" in message.lower() or "manual" in message.lower()
    assert "does not download" in message.lower()
    assert "separate" in message.lower()


def test_valid_schema_passes_validation():
    ieee_cis_validate.validate_transaction_dataset(_valid_df())  # no raise


def test_missing_required_columns_fails():
    df = _valid_df().drop(columns=["isFraud"])
    with pytest.raises(ieee_cis_validate.BenchmarkValidationError):
        ieee_cis_validate.validate_transaction_dataset(df)


def test_invalid_transaction_dt_fails():
    df = _valid_df()
    df.loc[0, "TransactionDT"] = -5
    with pytest.raises(ieee_cis_validate.BenchmarkValidationError):
        ieee_cis_validate.validate_transaction_dataset(df)


def test_non_numeric_transaction_dt_fails():
    df = _valid_df()
    df["TransactionDT"] = df["TransactionDT"].astype(str)
    with pytest.raises(ieee_cis_validate.BenchmarkValidationError):
        ieee_cis_validate.validate_transaction_dataset(df)


def test_invalid_transaction_amt_fails():
    df = _valid_df()
    df.loc[0, "TransactionAmt"] = -10.0
    with pytest.raises(ieee_cis_validate.BenchmarkValidationError):
        ieee_cis_validate.validate_transaction_dataset(df)


def test_invalid_is_fraud_fails():
    df = _valid_df()
    df.loc[0, "isFraud"] = 2
    with pytest.raises(ieee_cis_validate.BenchmarkValidationError):
        ieee_cis_validate.validate_transaction_dataset(df)


def test_class_prevalence_is_correct():
    df = _valid_df()
    assert ieee_cis_validate.class_prevalence(df) == pytest.approx(0.25)


def test_identity_columns_never_used_as_features():
    with pytest.raises(ieee_cis_validate.BenchmarkValidationError):
        ieee_cis_validate.assert_no_identity_columns_used(["transaction_amt", "id_01"])
    with pytest.raises(ieee_cis_validate.BenchmarkValidationError):
        ieee_cis_validate.assert_no_identity_columns_used(["TransactionID"])
    with pytest.raises(ieee_cis_validate.BenchmarkValidationError):
        ieee_cis_validate.assert_no_identity_columns_used(["isFraud"])
    ieee_cis_validate.assert_no_identity_columns_used(["transaction_amt", "product_cd"])  # no raise


def test_train_identity_is_never_loaded_or_required(tmp_path):
    """train_identity.csv must never be read by the loader -- confirmed by
    writing only train_transaction.csv to a temp dir and successfully
    loading it, with no train_identity.csv present anywhere."""

    dataset_path = tmp_path / "train_transaction.csv"
    _valid_df().to_csv(dataset_path, index=False)
    assert not (tmp_path / "train_identity.csv").exists()

    df = ieee_cis_loader.load_train_transaction(dataset_path)
    assert "train_identity" not in "".join(df.columns).lower()
    assert len(df) == 4


def test_loader_has_exactly_one_read_csv_call_and_it_is_not_identity_file():
    import inspect

    source = inspect.getsource(ieee_cis_loader)
    assert source.count("pd.read_csv(") == 1
    read_csv_line = next(line for line in source.splitlines() if "pd.read_csv(" in line)
    assert "identity" not in read_csv_line.lower()


def test_dataset_folder_is_gitignored():
    with open(".gitignore") as f:
        gitignore_content = f.read()
    assert "data/external/" in gitignore_content
