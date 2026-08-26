"""Milestone 3: train the Logistic Regression baseline for
label_high_loss_next_30d and select an operating threshold on validation data.

Preprocessing (imputation, scaling, one-hot encoding) is fit on the training
split only. The held-out test split is never touched by this script -- it is
not used for feature selection, scaling, model fitting, or threshold
selection. This is a synthetic-data demonstration model, not a real-world
chargeback prediction model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.features import LABEL_COLUMN, compute_feature_frame
from ml.model_utils import ML_FEATURE_COLUMNS, build_preprocessing_pipeline, compute_metrics
from ml.split_data import DEFAULT_CSV_PATH, load_and_split

DEFAULT_SEED = 42
DEFAULT_ARTIFACT_DIR = Path("ml/artifacts")
MODEL_VERSION = "0.1.0"

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
FBETA_BETA = 2.0
MIN_PRECISION = 0.30
MAX_FALSE_POSITIVE_RATE = 0.20


def _design_matrix(df):
    features = compute_feature_frame(df)
    return features[ML_FEATURE_COLUMNS]


def select_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> dict:
    """Selects the validation threshold maximizing F-beta(2), subject to
    precision >= MIN_PRECISION and false-positive rate <= MAX_FALSE_POSITIVE_RATE.
    Falls back to the unconstrained F-beta(2) maximizer, clearly flagged, if
    no threshold satisfies both constraints. Held-out test data is never
    used here.
    """
    candidates = []
    for t in THRESHOLD_GRID:
        preds = (val_probs >= t).astype(int)
        metrics = compute_metrics(y_val, preds, val_probs)
        candidates.append({"threshold": float(t), **metrics})

    constrained = [
        c for c in candidates
        if c["precision"] >= MIN_PRECISION and c["false_positive_rate"] <= MAX_FALSE_POSITIVE_RATE
    ]

    if constrained:
        best = max(constrained, key=lambda c: c["f2"])
        return {
            "selected_threshold": best["threshold"],
            "selection_method": "f2_maximizing_subject_to_constraints",
            "constraints_met": True,
            "min_precision_constraint": MIN_PRECISION,
            "max_fpr_constraint": MAX_FALSE_POSITIVE_RATE,
            "candidates": candidates,
            "selected_candidate_metrics": best,
        }

    fallback = max(candidates, key=lambda c: c["f2"])
    return {
        "selected_threshold": fallback["threshold"],
        "selection_method": "f2_maximizing_unconstrained_fallback",
        "constraints_met": False,
        "min_precision_constraint": MIN_PRECISION,
        "max_fpr_constraint": MAX_FALSE_POSITIVE_RATE,
        "candidates": candidates,
        "selected_candidate_metrics": fallback,
        "fallback_reason": (
            f"No threshold in the 0.05-0.95 grid met precision >= {MIN_PRECISION} AND "
            f"false-positive rate <= {MAX_FALSE_POSITIVE_RATE} simultaneously on validation data. "
            "Falling back to the threshold that maximizes F-beta(2) alone."
        ),
    }


def train(csv_path: Path = DEFAULT_CSV_PATH, seed: int = DEFAULT_SEED, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict:
    split = load_and_split(csv_path)
    train_df, val_df = split["train"], split["validation"]

    print("Final ML feature columns:", ML_FEATURE_COLUMNS)

    X_train = _design_matrix(train_df)
    y_train = train_df[LABEL_COLUMN].values
    X_val = _design_matrix(val_df)
    y_val = val_df[LABEL_COLUMN].values

    preprocessor = build_preprocessing_pipeline()
    model = LogisticRegression(max_iter=1000, random_state=seed)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    val_probs = pipeline.predict_proba(X_val)[:, 1]
    threshold_result = select_threshold(y_val, val_probs)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"logistic_regression_v{MODEL_VERSION}.joblib"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "seed": seed,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "feature_columns": ML_FEATURE_COLUMNS,
        "training_row_count": int(len(train_df)),
        "validation_row_count": int(len(val_df)),
        "split_info": split["split_info"],
        "selected_threshold": threshold_result["selected_threshold"],
        "threshold_selection_method": threshold_result["selection_method"],
        "threshold_constraints_met": threshold_result["constraints_met"],
        "min_precision_constraint": MIN_PRECISION,
        "max_fpr_constraint": MAX_FALSE_POSITIVE_RATE,
        "class_weight": None,
        "statement": (
            "This model is trained on synthetic data for demonstration and decision-support "
            "purposes only. It is not a real-world chargeback prediction model."
        ),
    }
    metadata_path = artifact_dir / f"logistic_regression_v{MODEL_VERSION}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Selected threshold: {threshold_result['selected_threshold']} ({threshold_result['selection_method']})")
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")

    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "threshold_result": threshold_result,
        "split": split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Milestone 3 Logistic Regression baseline.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    args = parser.parse_args()
    train(Path(args.csv_path), args.seed, Path(args.artifact_dir))


if __name__ == "__main__":
    main()
