import pandas as pd

from ml.generate_synthetic_data import generate_dataset
from ml.split_data import compute_split_weeks, split_dataset


def _dataset():
    return generate_dataset(seed=42, n_merchants=220, n_weeks=52)


def test_unique_weeks_never_overlap_between_splits():
    df = _dataset()
    result = split_dataset(df)
    train_weeks = set(result["train"]["week_start"].unique())
    val_weeks = set(result["validation"]["week_start"].unique())
    test_weeks = set(result["test"]["week_start"].unique())

    assert train_weeks.isdisjoint(val_weeks)
    assert train_weeks.isdisjoint(test_weeks)
    assert val_weeks.isdisjoint(test_weeks)


def test_splits_follow_chronological_order():
    df = _dataset()
    result = split_dataset(df)
    train_max = max(result["train"]["week_start"])
    val_min = min(result["validation"]["week_start"])
    val_max = max(result["validation"]["week_start"])
    test_min = min(result["test"]["week_start"])

    assert train_max < val_min
    assert val_max < test_min


def test_split_fractions_approximate_70_15_15():
    unique_weeks = sorted(set(pd.date_range("2025-01-06", periods=52, freq="7D").strftime("%Y-%m-%d")))
    train_weeks, val_weeks, test_weeks = compute_split_weeks(unique_weeks)

    assert len(train_weeks) + len(val_weeks) + len(test_weeks) == len(unique_weeks)
    assert abs(len(train_weeks) / len(unique_weeks) - 0.70) < 0.05
    assert abs(len(val_weeks) / len(unique_weeks) - 0.15) < 0.05
    assert abs(len(test_weeks) / len(unique_weeks) - 0.15) < 0.05


def test_no_row_shuffling_row_order_preserved_within_split():
    df = _dataset()
    result = split_dataset(df)
    train_df = result["train"]
    # Rows for a given merchant should remain in ascending week_start order.
    for merchant_id, group in train_df.groupby("merchant_id"):
        weeks = group["week_start"].tolist()
        assert weeks == sorted(weeks)


def test_split_info_reports_correct_row_and_week_counts():
    df = _dataset()
    result = split_dataset(df)
    info = result["split_info"]
    assert info["train"]["row_count"] == len(result["train"])
    assert info["validation"]["row_count"] == len(result["validation"])
    assert info["test"]["row_count"] == len(result["test"])
    assert info["train"]["row_count"] + info["validation"]["row_count"] + info["test"]["row_count"] == len(df)
