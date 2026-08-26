"""Chronological split, feature pipeline, and Logistic Regression baseline
for the IEEE-CIS external benchmark.

No random split, no shuffling across time, no secondary models (gradient
boosting, XGBoost, random forest, neural nets, SHAP, and hyperparameter
search are all out of scope for v1 -- see docs/EXTERNAL_BENCHMARK_DESIGN.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_SEED = 42

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
# Remaining ~0.15 is the held-out test fraction.

# Logistic Regression's built-in class_weight="balanced" is used as the
# documented class-imbalance handling for v1 -- simple, transparent, no
# resampling library or synthetic oversampling is introduced.
CLASS_WEIGHT = "balanced"

F2_BETA = 2.0
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by TransactionDT order -- earliest train_fraction rows to
    train, next validation_fraction to validation, remainder to held-out
    test. No shuffling; ties in TransactionDT are broken by original row
    order (stable sort)."""

    ordered = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * validation_fraction)

    train_df = ordered.iloc[:train_end]
    val_df = ordered.iloc[train_end:val_end]
    test_df = ordered.iloc[val_end:]
    return train_df, val_df, test_df


def describe_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    def _bounds(part: pd.DataFrame) -> dict:
        if len(part) == 0:
            return {"row_count": 0, "transaction_dt_min": None, "transaction_dt_max": None}
        return {
            "row_count": int(len(part)),
            "transaction_dt_min": float(part["TransactionDT"].min()),
            "transaction_dt_max": float(part["TransactionDT"].max()),
        }

    return {"train": _bounds(train_df), "validation": _bounds(val_df), "test": _bounds(test_df)}


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Missing-value handling and categorical encoding are fitted only on
    training data -- this Pipeline object is unfit until .fit() is called
    by the caller on the training split alone."""

    transformers = [("numeric", SimpleImputer(strategy="median"), numeric_features)]
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    model = LogisticRegression(class_weight=CLASS_WEIGHT, max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def select_threshold(y_val: pd.Series, probs_val: np.ndarray) -> dict:
    """Select an operating threshold on validation data only, maximizing
    F2 (recall-weighted, since missed fraud is typically costlier than a
    false alarm). Never uses held-out test data. Returns the full grid
    plus the selected threshold and its validation metrics."""

    grid_results = []
    best = None
    for threshold in THRESHOLD_GRID:
        preds = (probs_val >= threshold).astype(int)
        f2 = fbeta_score(y_val, preds, beta=F2_BETA, zero_division=0)
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        entry = {"threshold": float(threshold), "f2": float(f2), "precision": float(precision), "recall": float(recall)}
        grid_results.append(entry)
        if best is None or f2 > best["f2"]:
            best = entry

    if best is None or best["f2"] == 0.0:
        # Fallback: no threshold produced any positive F2 signal (e.g. the
        # model never separates the classes on this validation slice).
        # Documented, safe fallback rather than silently picking an
        # arbitrary threshold -- default to 0.5 and flag it explicitly.
        fallback_threshold = 0.5
        preds = (probs_val >= fallback_threshold).astype(int)
        best = {
            "threshold": fallback_threshold,
            "f2": float(fbeta_score(y_val, preds, beta=F2_BETA, zero_division=0)),
            "precision": float(precision_score(y_val, preds, zero_division=0)),
            "recall": float(recall_score(y_val, preds, zero_division=0)),
            "fallback_used": True,
        }
    else:
        best = {**best, "fallback_used": False}

    return {"grid": grid_results, "selected": best}
