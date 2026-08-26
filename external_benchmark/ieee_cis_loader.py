"""Local file loader for the IEEE-CIS Fraud Detection external benchmark.

This module never downloads data, never stores or requests Kaggle
credentials, and never reads train_identity.csv (deferred -- see
docs/EXTERNAL_BENCHMARK_DESIGN.md). It only reads a manually-placed local
CSV at the expected path and fails with a clear, safe message if it is
missing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_DATASET_PATH = Path("data/external/ieee_cis/train_transaction.csv")


def _manual_download_message(path: Path) -> str:
    return (
        "IEEE-CIS external benchmark dataset not found.\n\n"
        f"Expected local path: {path}\n\n"
        "This project does not download, verify, or redistribute this dataset. "
        "Before running this benchmark, you must manually and independently "
        "download the official IEEE-CIS Fraud Detection train_transaction.csv "
        "file from Kaggle, review its source terms/license/access conditions "
        f"yourself, and place it at exactly: {path}\n\n"
        "This benchmark is a separate, standalone transaction-level fraud "
        "classification experiment. It is not part of, and does not affect, "
        "ClearRisk Recover's core synthetic merchant-week refund/chargeback-loss "
        "model, rules engine, API, database, or dashboard."
    )


# Kept for backward-referencing the default path's message.
MANUAL_DOWNLOAD_MESSAGE = _manual_download_message(EXPECTED_DATASET_PATH)


class DatasetNotFoundError(Exception):
    """Raised when the manually-placed IEEE-CIS CSV is missing."""


def load_train_transaction(path: Path | str = EXPECTED_DATASET_PATH) -> pd.DataFrame:
    """Load train_transaction.csv from a local path. Never loads
    train_identity.csv and never attempts any join -- that dataset is
    deferred out of v1 scope entirely."""

    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(_manual_download_message(path))
    return pd.read_csv(path)


def preview(df: pd.DataFrame, n_rows: int = 3) -> str:
    """Return a safe shape/schema-only preview -- explicitly requested by
    the caller, never printed automatically. Shows column names, dtypes,
    and the first n_rows values, which is acceptable here because this
    dataset is already a public, anonymized competition dataset (no real
    PII), unlike ClearRisk's own synthetic-data-only restrictions."""

    lines = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns", "Dtypes:", str(df.dtypes), f"First {n_rows} rows:", str(df.head(n_rows))]
    return "\n".join(lines)
