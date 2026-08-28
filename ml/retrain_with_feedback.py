"""Manually-triggered feedback retraining loop (Phase 2).

Never runs automatically -- always invoked explicitly via
`python3 -m ml.retrain_with_feedback`, and only after
`python3 scripts/export_feedback_labels.py` has produced a fresh
label-override file from reviewer resolutions.

Label corrections (FALSE_POSITIVE -> 0, CONFIRMED_RISK -> 1) are applied
ONLY to rows in the time-based training split. The held-out test split is
never touched, so before/after metrics on it remain a fair, uncorrupted
comparison -- the whole point of having a held-out set in the first place.

This module has no dependency on app/ (mirrors the rest of ml/) -- it
consumes the JSON file scripts/export_feedback_labels.py produces, it does
not query the database itself.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.evaluate_model import evaluate_split
from ml.features import LABEL_COLUMN, compute_feature_frame
from ml.model_utils import ML_FEATURE_COLUMNS, build_preprocessing_pipeline
from ml.split_data import DEFAULT_CSV_PATH, load_and_split
from ml.train_baseline_model import DEFAULT_SEED, select_threshold

DEFAULT_FEEDBACK_PATH = Path("ml/artifacts/feedback_label_overrides.json")
DEFAULT_ARTIFACT_DIR = Path("ml/artifacts")
BASELINE_MODEL_VERSION = "0.1.0"
FEEDBACK_MODEL_VERSION = "0.1.0-feedback1"


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return compute_feature_frame(df)[ML_FEATURE_COLUMNS]


def load_feedback_overrides(path: Path = DEFAULT_FEEDBACK_PATH) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_feedback_to_training_split(train_df: pd.DataFrame, overrides: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Applies label corrections only to rows that exist in train_df (i.e.
    only overrides whose week falls in the training split). Returns
    (corrected_df, application_report) -- every override is accounted for,
    either as applied or skipped-with-a-reason. Nothing is silently
    dropped."""
    corrected = train_df.copy()
    applied: list[dict] = []
    skipped: list[dict] = []

    for override in overrides:
        mask = (corrected["merchant_id"] == override["merchant_id"]) & (corrected["week_start"] == override["week_start"])
        if not mask.any():
            skipped.append({
                **override,
                "reason": "No matching row in the training split (the week is in validation/test, or merchant_id/week_start didn't match any row).",
            })
            continue
        original_label = int(corrected.loc[mask, LABEL_COLUMN].iloc[0])
        if original_label == override["corrected_label"]:
            skipped.append({**override, "reason": "Label already matches the corrected value -- no change needed."})
            continue
        corrected.loc[mask, LABEL_COLUMN] = override["corrected_label"]
        applied.append({**override, "original_label": original_label})

    return corrected, {"applied": applied, "skipped": skipped}


def retrain_with_feedback(
    csv_path: Path = DEFAULT_CSV_PATH,
    feedback_path: Path = DEFAULT_FEEDBACK_PATH,
    seed: int = DEFAULT_SEED,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> dict:
    split = load_and_split(csv_path)
    train_df, val_df, test_df = split["train"], split["validation"], split["test"]

    overrides = load_feedback_overrides(feedback_path)
    corrected_train_df, application_report = apply_feedback_to_training_split(train_df, overrides)

    X_train = _design_matrix(corrected_train_df)
    y_train = corrected_train_df[LABEL_COLUMN].values
    X_val = _design_matrix(val_df)
    y_val = val_df[LABEL_COLUMN].values

    preprocessor = build_preprocessing_pipeline()
    model = LogisticRegression(max_iter=1000, random_state=seed)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    val_probs = pipeline.predict_proba(X_val)[:, 1]
    threshold_result = select_threshold(y_val, val_probs)
    new_threshold = threshold_result["selected_threshold"]

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"logistic_regression_v{FEEDBACK_MODEL_VERSION}.joblib"
    joblib.dump(pipeline, model_path)

    # Load the existing baseline artifact for a true before/after
    # comparison, evaluated on the identical, untouched held-out test split.
    baseline_model_path = artifact_dir / f"logistic_regression_v{BASELINE_MODEL_VERSION}.joblib"
    baseline_metadata_path = artifact_dir / f"logistic_regression_v{BASELINE_MODEL_VERSION}_metadata.json"
    if not baseline_model_path.exists() or not baseline_metadata_path.exists():
        raise FileNotFoundError(
            f"No baseline model found at {baseline_model_path}. Run `python3 -m ml.train_baseline_model` first."
        )
    baseline_pipeline = joblib.load(baseline_model_path)
    with open(baseline_metadata_path, "r", encoding="utf-8") as f:
        baseline_metadata = json.load(f)
    baseline_threshold = baseline_metadata["selected_threshold"]

    before_test_results = evaluate_split(test_df, baseline_pipeline, baseline_threshold)
    after_test_results = evaluate_split(test_df, pipeline, new_threshold)

    report = {
        "report_version": "1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "synthetic_data_notice": (
            "Synthetic-data feedback retraining demonstration only. Feedback comes from "
            "simulated reviewer resolutions on synthetic cases, not real-world outcomes. "
            "This retrain was manually triggered -- no automatic retraining exists anywhere "
            "in this codebase."
        ),
        "feedback_application": application_report,
        "feedback_summary": {
            "total_overrides_available": len(overrides),
            "applied_to_training": len(application_report["applied"]),
            "skipped": len(application_report["skipped"]),
        },
        "baseline_model_version": BASELINE_MODEL_VERSION,
        "feedback_model_version": FEEDBACK_MODEL_VERSION,
        "baseline_threshold": baseline_threshold,
        "feedback_threshold": new_threshold,
        "held_out_test_comparison": {
            "note": "Both evaluated on the identical, untouched held-out test split -- only training-split labels were corrected.",
            "before_feedback": before_test_results,
            "after_feedback": after_test_results,
        },
    }

    report_path = artifact_dir / "feedback_retrain_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Applied {len(application_report['applied'])} feedback override(s) to the training split "
          f"({len(application_report['skipped'])} skipped -- see the report for reasons).")
    print(f"Feedback-retrained model saved to {model_path}")
    print(f"Before/after comparison report saved to {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually retrain the Logistic Regression baseline using reviewer feedback labels. Never runs automatically."
    )
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--feedback-path", type=str, default=str(DEFAULT_FEEDBACK_PATH))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    args = parser.parse_args()
    retrain_with_feedback(Path(args.csv_path), Path(args.feedback_path), args.seed, Path(args.artifact_dir))


if __name__ == "__main__":
    main()
