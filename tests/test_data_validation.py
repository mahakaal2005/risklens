import pandas as pd
import pytest

from ml.data_validation import DataValidationError, validate_dataset
from ml.generate_synthetic_data import generate_dataset


@pytest.fixture(scope="module")
def valid_df():
    return generate_dataset(seed=42, n_merchants=220, n_weeks=52)


def test_valid_dataset_passes(valid_df):
    validate_dataset(valid_df)  # should not raise


def test_missing_column_is_rejected(valid_df):
    df = valid_df.drop(columns=["refund_rate_30d"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataset(df)


def test_out_of_range_rate_is_rejected(valid_df):
    df = valid_df.copy()
    df.loc[0, "refund_rate_30d"] = 1.5
    with pytest.raises(DataValidationError, match=r"refund_rate_30d.*outside \[0, 1\]"):
        validate_dataset(df)


def test_negative_numeric_field_is_rejected(valid_df):
    df = valid_df.copy()
    df.loc[0, "transaction_count_30d"] = -5
    with pytest.raises(DataValidationError, match="negative values"):
        validate_dataset(df)


def test_invalid_label_value_is_rejected(valid_df):
    df = valid_df.copy()
    df.loc[0, "label_high_loss_next_30d"] = 2
    with pytest.raises(DataValidationError, match="label_high_loss_next_30d contains values other than 0/1"):
        validate_dataset(df)


def test_duplicate_merchant_week_is_rejected(valid_df):
    df = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate merchant_id"):
        validate_dataset(df)


def test_non_synthetic_merchant_id_is_rejected(valid_df):
    df = valid_df.copy()
    df.loc[0, "merchant_id"] = "real_merchant_12345"
    with pytest.raises(DataValidationError, match="not synthetic tokens"):
        validate_dataset(df)


def test_unparseable_week_start_is_rejected(valid_df):
    df = valid_df.copy()
    df.loc[0, "week_start"] = "not-a-date"
    with pytest.raises(DataValidationError, match="do not parse as a date"):
        validate_dataset(df)


def test_missing_latent_state_is_rejected(valid_df):
    df = valid_df[valid_df["latent_state_for_demo_only"] != "early_hidden_risk"].copy()
    with pytest.raises(DataValidationError, match="Missing required latent state"):
        validate_dataset(df)


def test_single_label_class_is_rejected(valid_df):
    df = valid_df.copy()
    df["label_high_loss_next_30d"] = 0
    with pytest.raises(DataValidationError, match="only one label class"):
        validate_dataset(df)


def test_prohibited_field_is_rejected(valid_df):
    df = valid_df.copy()
    df["email"] = "demo@example.com"
    with pytest.raises(DataValidationError, match="Prohibited fields present"):
        validate_dataset(df)


def test_missing_false_positive_opportunities_is_rejected(valid_df):
    df = valid_df.copy()
    seasonal_mask = df["latent_state_for_demo_only"] == "seasonal_sale_legitimate_returns"
    df.loc[seasonal_mask, "label_high_loss_next_30d"] = 1
    with pytest.raises(DataValidationError, match="No seasonal-sale false-positive opportunities"):
        validate_dataset(df)


def test_missing_false_negative_opportunities_is_rejected(valid_df):
    df = valid_df.copy()
    early_hidden_mask = df["latent_state_for_demo_only"] == "early_hidden_risk"
    df.loc[early_hidden_mask, "label_high_loss_next_30d"] = 0
    with pytest.raises(DataValidationError, match="No early-hidden-risk false-negative opportunities"):
        validate_dataset(df)
