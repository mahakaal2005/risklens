"""Schema and honesty validation for the IEEE-CIS external benchmark input.

Mirrors ml/data_validation.py's "fail safely, never repair or drop rows"
principle, adapted for this external, transaction-level dataset. Never
requires or loads train_identity.csv.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["TransactionDT", "TransactionAmt", "isFraud"]

# Columns that belong to the deferred identity table or that would require
# joining it. Never used as features in v1, even if a column with this
# name were ever present in train_transaction.csv.
IDENTITY_COLUMN_PREFIXES = ("id_", "DeviceType", "DeviceInfo")

# Columns that are raw record identifiers, not features.
ID_LIKE_COLUMNS = ("TransactionID",)


class BenchmarkValidationError(Exception):
    """Raised when the IEEE-CIS input fails schema or honesty validation.
    Aggregation/training never proceeds past this -- bad rows are never
    silently dropped or repaired."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("IEEE-CIS benchmark input validation failed:\n- " + "\n- ".join(issues))


def validate_transaction_dataset(df: pd.DataFrame) -> None:
    """Validate a train_transaction.csv-shaped DataFrame. Raises
    BenchmarkValidationError listing every problem found."""

    issues: list[str] = []

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
        raise BenchmarkValidationError(issues)

    if not pd.api.types.is_numeric_dtype(df["TransactionDT"]):
        issues.append("TransactionDT is not numeric")
    elif (df["TransactionDT"] < 0).any():
        issues.append("TransactionDT contains negative values")

    if not pd.api.types.is_numeric_dtype(df["TransactionAmt"]):
        issues.append("TransactionAmt is not numeric")
    elif (df["TransactionAmt"] < 0).any():
        issues.append("TransactionAmt contains negative values")

    if not pd.api.types.is_numeric_dtype(df["isFraud"]):
        issues.append("isFraud is not numeric")
    else:
        label_values = set(df["isFraud"].dropna().unique().tolist())
        if not label_values.issubset({0, 1}):
            issues.append(f"isFraud contains values other than 0/1: {label_values}")

    if issues:
        raise BenchmarkValidationError(issues)


def class_prevalence(df: pd.DataFrame) -> float:
    """Positive (isFraud == 1) rate. Caller must validate first."""
    return float(df["isFraud"].mean())


def assert_no_identity_columns_used(columns: list[str]) -> None:
    """Defensive guard: raise if any identity-table or ID-like column name
    appears in a proposed feature list. Called by ieee_cis_features.py
    before returning its selected feature list."""

    violations = [
        c
        for c in columns
        if c.startswith(IDENTITY_COLUMN_PREFIXES) or c in ID_LIKE_COLUMNS or c == "isFraud"
    ]
    if violations:
        raise BenchmarkValidationError(
            [f"Identity-table, ID-like, or target column(s) proposed as feature(s): {violations}"]
        )
