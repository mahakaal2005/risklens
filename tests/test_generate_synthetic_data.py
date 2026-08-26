import pandas as pd

from ml.data_validation import (
    REQUIRED_LATENT_STATES,
    count_early_hidden_risk_false_negative_opportunities,
    count_seasonal_false_positive_opportunities,
    validate_dataset,
)
from ml.generate_synthetic_data import generate_dataset

SMALL_N_MERCHANTS = 60
SMALL_N_WEEKS = 30


def _small_dataset(seed: int = 42) -> pd.DataFrame:
    return generate_dataset(seed=seed, n_merchants=SMALL_N_MERCHANTS, n_weeks=SMALL_N_WEEKS)


def test_generation_is_reproducible_with_same_seed():
    df1 = _small_dataset(seed=42)
    df2 = _small_dataset(seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_create_different_datasets():
    df1 = _small_dataset(seed=42)
    df2 = _small_dataset(seed=43)
    assert not df1.equals(df2)


def test_required_columns_exist():
    df = _small_dataset()
    expected = {
        "merchant_id", "week_start", "merchant_category", "merchant_age_days",
        "transaction_count_30d", "transaction_volume_30d", "transaction_volume_previous_30d",
        "transaction_volume_change_30d", "refund_count_30d", "refund_rate_30d",
        "refund_rate_previous_30d", "refund_rate_change_30d", "chargeback_count_30d",
        "chargeback_rate_30d", "chargeback_rate_previous_30d", "chargeback_rate_change_30d",
        "top_dispute_reason_category", "delivery_evidence_coverage", "support_ticket_rate",
        "average_support_resolution_time_hours", "previous_review_outcome",
        "label_high_loss_next_30d", "latent_state_for_demo_only",
    }
    assert expected.issubset(set(df.columns))


def test_dataset_has_requested_merchant_and_week_count():
    df = _small_dataset()
    assert df["merchant_id"].nunique() == SMALL_N_MERCHANTS
    assert len(df) == SMALL_N_MERCHANTS * SMALL_N_WEEKS


def test_no_duplicate_merchant_week_rows():
    df = _small_dataset()
    assert not df.duplicated(subset=["merchant_id", "week_start"]).any()


def test_all_five_latent_states_appear():
    df = generate_dataset(seed=42, n_merchants=220, n_weeks=52)
    present = set(df["latent_state_for_demo_only"].unique())
    assert set(REQUIRED_LATENT_STATES).issubset(present)


def test_both_label_classes_appear():
    df = _small_dataset()
    assert set(df["label_high_loss_next_30d"].unique()) == {0, 1}


def test_no_prohibited_sensitive_fields():
    df = _small_dataset()
    prohibited = {"card_pan", "card_number", "cvv", "upi_pin", "bank_password", "aadhaar_number", "email", "phone_number"}
    assert prohibited.isdisjoint(set(df.columns))


def test_rate_fields_are_valid_ranges():
    df = generate_dataset(seed=42, n_merchants=220, n_weeks=52)
    for col in ["refund_rate_30d", "refund_rate_previous_30d", "chargeback_rate_30d", "chargeback_rate_previous_30d", "delivery_evidence_coverage"]:
        assert df[col].between(0, 1).all(), f"{col} out of [0,1] range"


def test_full_dataset_passes_validation():
    df = generate_dataset(seed=42, n_merchants=220, n_weeks=52)
    validate_dataset(df)  # should not raise


def test_seasonal_sale_cases_include_label_zero_despite_elevated_refunds():
    df = generate_dataset(seed=42, n_merchants=220, n_weeks=52)
    assert count_seasonal_false_positive_opportunities(df) > 0


def test_early_hidden_risk_cases_include_label_one_despite_modest_signals():
    df = generate_dataset(seed=42, n_merchants=220, n_weeks=52)
    assert count_early_hidden_risk_false_negative_opportunities(df) > 0
