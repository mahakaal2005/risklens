"""Train Random Forest and Gradient Boosting as additional held-out-evaluated
comparison baselines next to the Logistic Regression model trained by
ml/train_baseline_model.py.

Design decision (see the 2026-08-28 model-comparison plan): these two models
are evaluation-only. Logistic Regression remains the sole model used for
live case scoring (ml/case_packet.py, app/services/case_service.py,
combined_policy in ml/model_utils.py) -- this keeps CLAUDE.md's "prefer
transparent rules and interpretable models before complex models" principle
intact for anything user-facing, while still answering honestly whether
more complex models actually do better on this synthetic held-out test set.

Same preprocessing, same time-based split, same threshold-selection method
as the Logistic Regression baseline -- the only thing that differs between
these three training scripts is the model class itself, so any accuracy
difference reflects the model, not an uneven evaluation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml.features import LABEL_COLUMN, compute_feature_frame
from ml.model_utils import MAX_FALSE_POSITIVE_RATE, MIN_PRECISION, ML_FEATURE_COLUMNS, build_preprocessing_pipeline, select_threshold
from ml.split_data import DEFAULT_CSV_PATH, load_and_split

DEFAULT_SEED = 42
DEFAULT_ARTIFACT_DIR = Path("ml/artifacts")

# name -> (artifact stem, model version, sklearn estimator factory)
TREE_MODELS = {
    "random_forest": {
        "artifact_stem": "random_forest",
        "model_version": "0.1.0",
        "display_name": "Random Forest",
        "build_estimator": lambda seed: RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    },
    "gradient_boosting": {
        "artifact_stem": "gradient_boosting",
        "model_version": "0.1.0",
        "display_name": "Gradient Boosting",
        "build_estimator": lambda seed: HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            random_state=seed,
        ),
    },
}


def _design_matrix(df):
    features = compute_feature_frame(df)
    return features[ML_FEATURE_COLUMNS]


def train_one(model_key: str, csv_path: Path = DEFAULT_CSV_PATH, seed: int = DEFAULT_SEED, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict:
    spec = TREE_MODELS[model_key]
    split = load_and_split(csv_path)
    train_df, val_df = split["train"], split["validation"]

    X_train = _design_matrix(train_df)
    y_train = train_df[LABEL_COLUMN].values
    X_val = _design_matrix(val_df)
    y_val = val_df[LABEL_COLUMN].values

    preprocessor = build_preprocessing_pipeline()
    model = spec["build_estimator"](seed)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    val_probs = pipeline.predict_proba(X_val)[:, 1]
    threshold_result = select_threshold(y_val, val_probs)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{spec['artifact_stem']}_v{spec['model_version']}.joblib"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": spec["display_name"],
        "model_version": spec["model_version"],
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
        "role": "evaluation_comparison_only",
        "statement": (
            "This model is trained on synthetic data for demonstration and decision-support "
            "purposes only. It is not a real-world chargeback prediction model. It is evaluated "
            "as a comparison baseline only -- Logistic Regression remains the model used for "
            "live case scoring, per CLAUDE.md's preference for interpretable models."
        ),
    }
    metadata_path = artifact_dir / f"{spec['artifact_stem']}_v{spec['model_version']}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[{spec['display_name']}] selected threshold: {threshold_result['selected_threshold']} ({threshold_result['selection_method']})")
    print(f"[{spec['display_name']}] model saved to {model_path}")
    print(f"[{spec['display_name']}] metadata saved to {metadata_path}")

    return {"pipeline": pipeline, "metadata": metadata, "threshold_result": threshold_result, "split": split}


def train_all(csv_path: Path = DEFAULT_CSV_PATH, seed: int = DEFAULT_SEED, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict:
    return {model_key: train_one(model_key, csv_path, seed, artifact_dir) for model_key in TREE_MODELS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Random Forest and Gradient Boosting comparison baselines.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    args = parser.parse_args()
    train_all(Path(args.csv_path), args.seed, Path(args.artifact_dir))


if __name__ == "__main__":
    main()
