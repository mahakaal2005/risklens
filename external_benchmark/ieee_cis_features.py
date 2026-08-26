"""Compact, explainable, transaction-only feature set for the IEEE-CIS
external benchmark. Deliberately narrow for v1 -- see
docs/EXTERNAL_BENCHMARK_DESIGN.md Section 3 for what was left out and why.

Never uses train_identity.csv, target-leakage fields, direct identity
fields, or raw record identifiers as features.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from external_benchmark.ieee_cis_validate import assert_no_identity_columns_used

SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600

REQUIRED_NUMERIC_FEATURES = [
    "transaction_amt",
    "transaction_amt_has_cents",
    "transaction_relative_day",
    "transaction_hour_of_day_approx",
]

# Only used if present in the input file -- ProductCD is a documented,
# non-identity, transaction-level categorical field in the public IEEE-CIS
# schema description.
OPTIONAL_CATEGORICAL_FEATURES = ["product_cd"]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the v1 feature frame from a validated train_transaction.csv
    -shaped DataFrame. Does not fit or apply any imputer/encoder -- that
    happens inside the sklearn Pipeline in ieee_cis_train.py, fitted only
    on the training split."""

    features = pd.DataFrame(index=df.index)
    features["transaction_amt"] = df["TransactionAmt"]
    features["transaction_amt_has_cents"] = (
        (df["TransactionAmt"] % 1 != 0).astype(int)
    )
    features["transaction_relative_day"] = (df["TransactionDT"] // SECONDS_PER_DAY).astype("Int64")
    features["transaction_hour_of_day_approx"] = ((df["TransactionDT"] // SECONDS_PER_HOUR) % 24).astype("Int64")

    if "ProductCD" in df.columns:
        features["product_cd"] = df["ProductCD"]

    selected_columns = list(features.columns)
    assert_no_identity_columns_used(selected_columns)
    return features


def selected_feature_columns(feature_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split a feature frame's columns into (numeric, categorical) lists
    for use by the sklearn Pipeline's ColumnTransformer."""

    categorical = [c for c in OPTIONAL_CATEGORICAL_FEATURES if c in feature_frame.columns]
    numeric = [c for c in feature_frame.columns if c not in categorical]
    return numeric, categorical


def save_feature_list(feature_frame: pd.DataFrame, path: Path | str) -> dict:
    """Print and persist the final selected feature list, as required by
    the approved scope. Returns the same dict that was written."""

    numeric, categorical = selected_feature_columns(feature_frame)
    payload = {"numeric_features": numeric, "categorical_features": categorical}
    print(f"Selected feature list: {json.dumps(payload, indent=2)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return payload
