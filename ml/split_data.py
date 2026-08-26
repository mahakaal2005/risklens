"""Strict chronological time-based split for the merchant-week dataset.

Splits by unique ordered week_start values -- never by shuffled rows -- so
a given week exists in exactly one of train/validation/test. Held-out test
data is never used for feature selection, scaling, model fitting, or
threshold selection; validation data may be used for model choice and
threshold selection only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CSV_PATH = Path("demo_data/synthetic_merchant_week_data.csv")

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
# Remaining ~0.15 is test.


def compute_split_weeks(sorted_unique_weeks: list[str]) -> tuple[list[str], list[str], list[str]]:
    n = len(sorted_unique_weeks)
    train_end = int(round(n * TRAIN_FRACTION))
    val_end = int(round(n * (TRAIN_FRACTION + VALIDATION_FRACTION)))
    train_weeks = sorted_unique_weeks[:train_end]
    val_weeks = sorted_unique_weeks[train_end:val_end]
    test_weeks = sorted_unique_weeks[val_end:]
    return train_weeks, val_weeks, test_weeks


def split_dataset(df: pd.DataFrame) -> dict:
    sorted_unique_weeks = sorted(df["week_start"].unique().tolist())
    train_weeks, val_weeks, test_weeks = compute_split_weeks(sorted_unique_weeks)

    train_df = df[df["week_start"].isin(train_weeks)].reset_index(drop=True)
    val_df = df[df["week_start"].isin(val_weeks)].reset_index(drop=True)
    test_df = df[df["week_start"].isin(test_weeks)].reset_index(drop=True)

    split_info = {
        "train": {
            "week_count": len(train_weeks),
            "row_count": int(len(train_df)),
            "week_start_min": train_weeks[0] if train_weeks else None,
            "week_start_max": train_weeks[-1] if train_weeks else None,
        },
        "validation": {
            "week_count": len(val_weeks),
            "row_count": int(len(val_df)),
            "week_start_min": val_weeks[0] if val_weeks else None,
            "week_start_max": val_weeks[-1] if val_weeks else None,
        },
        "test": {
            "week_count": len(test_weeks),
            "row_count": int(len(test_df)),
            "week_start_min": test_weeks[0] if test_weeks else None,
            "week_start_max": test_weeks[-1] if test_weeks else None,
        },
    }
    return {"train": train_df, "validation": val_df, "test": test_df, "split_info": split_info}


def load_and_split(csv_path: Path = DEFAULT_CSV_PATH) -> dict:
    df = pd.read_csv(csv_path)
    return split_dataset(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the chronological train/validation/test split summary.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH))
    args = parser.parse_args()

    result = load_and_split(Path(args.csv_path))
    for split_name, info in result["split_info"].items():
        print(f"{split_name}: {info['week_count']} weeks, {info['row_count']} rows, {info['week_start_min']} to {info['week_start_max']}")


if __name__ == "__main__":
    main()
