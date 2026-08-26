"""Validation for the synthetic merchant-week dataset.

Validates a whole dataset (not just one row) against the schema and honesty
requirements in DATA_DICTIONARY.md and MODEL_CARD.md. Malformed or dishonest
data is rejected with a clear error listing every problem found, per
CLAUDE.md's "fail safely" principle -- this module never silently repairs
or drops invalid rows.
"""

from __future__ import annotations

import re

import pandas as pd

REQUIRED_COLUMNS = [
    "merchant_id",
    "week_start",
    "merchant_category",
    "merchant_age_days",
    "transaction_count_30d",
    "transaction_volume_30d",
    "transaction_volume_previous_30d",
    "transaction_volume_change_30d",
    "refund_count_30d",
    "refund_rate_30d",
    "refund_rate_previous_30d",
    "refund_rate_change_30d",
    "chargeback_count_30d",
    "chargeback_rate_30d",
    "chargeback_rate_previous_30d",
    "chargeback_rate_change_30d",
    "top_dispute_reason_category",
    "delivery_evidence_coverage",
    "support_ticket_rate",
    "average_support_resolution_time_hours",
    "previous_review_outcome",
    "label_high_loss_next_30d",
    "latent_state_for_demo_only",
]

REQUIRED_LATENT_STATES = [
    "stable_merchant",
    "seasonal_sale_legitimate_returns",
    "operational_fulfilment_failure",
    "high_risk_merchant_behaviour",
    "early_hidden_risk",
]

NON_NEGATIVE_COLUMNS = [
    "merchant_age_days",
    "transaction_count_30d",
    "transaction_volume_30d",
    "transaction_volume_previous_30d",
    "refund_count_30d",
    "chargeback_count_30d",
    "support_ticket_rate",
    "average_support_resolution_time_hours",
]

RATE_COLUMNS_0_1 = [
    "refund_rate_30d",
    "refund_rate_previous_30d",
    "chargeback_rate_30d",
    "chargeback_rate_previous_30d",
    "delivery_evidence_coverage",
]

MERCHANT_ID_PATTERN = re.compile(r"^merchant_demo_\d+$")

# Field names/patterns that must never appear in synthetic data, per
# DATA_DICTIONARY.md "Prohibited fields" and SECURITY.md.
PROHIBITED_FIELD_NAMES = frozenset(
    {
        "card_pan",
        "card_number",
        "cvv",
        "upi_pin",
        "bank_password",
        "bank_login_token",
        "aadhaar_number",
        "pan_number",
        "account_number",
        "ifsc",
        "phone_number",
        "email",
        "address",
        "biometric",
        "device_fingerprint",
        "customer_name",
        "merchant_name",
    }
)

MIN_PLAUSIBLE_LABEL_RATE = 0.01
MAX_PLAUSIBLE_LABEL_RATE = 0.60

SEASONAL_STATE = "seasonal_sale_legitimate_returns"
EARLY_HIDDEN_STATE = "early_hidden_risk"
SEASONAL_ELEVATED_REFUND_THRESHOLD = 0.03
EARLY_HIDDEN_MODEST_REFUND_THRESHOLD = 0.04
EARLY_HIDDEN_MODEST_CHARGEBACK_THRESHOLD = 0.015


class DataValidationError(Exception):
    """Raised when the dataset fails schema or honesty validation."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Data validation failed:\n- " + "\n- ".join(issues))


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate a merchant-week dataframe. Raises DataValidationError listing
    every problem found; does not attempt to repair or drop rows."""

    issues: list[str] = []

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")

    prohibited_present = PROHIBITED_FIELD_NAMES & set(df.columns)
    if prohibited_present:
        issues.append(f"Prohibited fields present: {sorted(prohibited_present)}")

    if missing_columns:
        # Cannot safely run the remaining checks without the columns they need.
        raise DataValidationError(issues)

    bad_merchant_ids = df.loc[~df["merchant_id"].astype(str).str.match(MERCHANT_ID_PATTERN), "merchant_id"].unique()
    if len(bad_merchant_ids) > 0:
        issues.append(f"merchant_id values are not synthetic tokens matching '{MERCHANT_ID_PATTERN.pattern}': {list(bad_merchant_ids)[:5]}")

    parsed_dates = pd.to_datetime(df["week_start"], format="%Y-%m-%d", errors="coerce")
    if parsed_dates.isna().any():
        bad_count = int(parsed_dates.isna().sum())
        issues.append(f"{bad_count} week_start value(s) do not parse as a date")

    for col in NON_NEGATIVE_COLUMNS:
        if (df[col] < 0).any():
            issues.append(f"Column '{col}' contains negative values")

    for col in RATE_COLUMNS_0_1:
        if ((df[col] < 0) | (df[col] > 1)).any():
            issues.append(f"Column '{col}' contains values outside [0, 1]")

    label_values = set(df["label_high_loss_next_30d"].unique().tolist())
    if not label_values.issubset({0, 1}):
        issues.append(f"label_high_loss_next_30d contains values other than 0/1: {label_values}")

    duplicate_mask = df.duplicated(subset=["merchant_id", "week_start"], keep=False)
    if duplicate_mask.any():
        issues.append(f"{int(duplicate_mask.sum())} duplicate merchant_id + week_start row(s) found")

    present_states = set(df["latent_state_for_demo_only"].unique().tolist())
    missing_states = set(REQUIRED_LATENT_STATES) - present_states
    if missing_states:
        issues.append(f"Missing required latent state(s) in output: {sorted(missing_states)}")

    if label_values.issubset({0, 1}) and len(label_values) < 2:
        issues.append("Dataset contains only one label class; both 0 and 1 must be present")

    if label_values.issubset({0, 1}) and len(label_values) == 2:
        positive_rate = float(df["label_high_loss_next_30d"].mean())
        if positive_rate < MIN_PLAUSIBLE_LABEL_RATE or positive_rate > MAX_PLAUSIBLE_LABEL_RATE:
            issues.append(
                f"label_high_loss_next_30d positive rate {positive_rate:.4f} is outside the "
                f"plausible range [{MIN_PLAUSIBLE_LABEL_RATE}, {MAX_PLAUSIBLE_LABEL_RATE}]"
            )

    if SEASONAL_STATE in present_states:
        seasonal_false_positive_opportunities = df[
            (df["latent_state_for_demo_only"] == SEASONAL_STATE)
            & (df["label_high_loss_next_30d"] == 0)
            & (df["refund_rate_30d"] > SEASONAL_ELEVATED_REFUND_THRESHOLD)
        ]
        if len(seasonal_false_positive_opportunities) == 0:
            issues.append(
                "No seasonal-sale false-positive opportunities found "
                f"(latent_state={SEASONAL_STATE}, label=0, refund_rate_30d > {SEASONAL_ELEVATED_REFUND_THRESHOLD})"
            )

    if EARLY_HIDDEN_STATE in present_states:
        early_hidden_false_negative_opportunities = df[
            (df["latent_state_for_demo_only"] == EARLY_HIDDEN_STATE)
            & (df["label_high_loss_next_30d"] == 1)
            & (df["refund_rate_30d"] < EARLY_HIDDEN_MODEST_REFUND_THRESHOLD)
            & (df["chargeback_rate_30d"] < EARLY_HIDDEN_MODEST_CHARGEBACK_THRESHOLD)
        ]
        if len(early_hidden_false_negative_opportunities) == 0:
            issues.append(
                "No early-hidden-risk false-negative opportunities found "
                f"(latent_state={EARLY_HIDDEN_STATE}, label=1, refund_rate_30d < {EARLY_HIDDEN_MODEST_REFUND_THRESHOLD}, "
                f"chargeback_rate_30d < {EARLY_HIDDEN_MODEST_CHARGEBACK_THRESHOLD})"
            )

    if issues:
        raise DataValidationError(issues)


def count_seasonal_false_positive_opportunities(df: pd.DataFrame) -> int:
    return int(
        len(
            df[
                (df["latent_state_for_demo_only"] == SEASONAL_STATE)
                & (df["label_high_loss_next_30d"] == 0)
                & (df["refund_rate_30d"] > SEASONAL_ELEVATED_REFUND_THRESHOLD)
            ]
        )
    )


def count_early_hidden_risk_false_negative_opportunities(df: pd.DataFrame) -> int:
    return int(
        len(
            df[
                (df["latent_state_for_demo_only"] == EARLY_HIDDEN_STATE)
                & (df["label_high_loss_next_30d"] == 1)
                & (df["refund_rate_30d"] < EARLY_HIDDEN_MODEST_REFUND_THRESHOLD)
                & (df["chargeback_rate_30d"] < EARLY_HIDDEN_MODEST_CHARGEBACK_THRESHOLD)
            ]
        )
    )
