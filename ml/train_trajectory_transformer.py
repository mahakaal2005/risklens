"""Train the merchant risk trajectory Transformer as an additional
held-out-evaluated comparison baseline.

Same time-based split, same threshold-selection method, and same honesty rules
as ml/train_baseline_model.py and ml/train_tree_models.py -- the only thing that
differs is the model class, so any difference in the reported metrics reflects
the model rather than an uneven evaluation.

Like Random Forest and Gradient Boosting, this model is evaluation-only:
Logistic Regression remains the model used for live case scoring.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
import torch

from ml.features import LABEL_COLUMN
from ml.model_utils import MAX_FALSE_POSITIVE_RATE, MIN_PRECISION, ML_FEATURE_COLUMNS, select_threshold
from ml.split_data import DEFAULT_CSV_PATH, load_and_split
from ml.trajectory_transformer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_D_MODEL,
    DEFAULT_EPOCHS,
    DEFAULT_LAYERS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_N_HEADS,
    DEFAULT_WINDOW,
    TrajectoryModel,
)

DEFAULT_SEED = 42
DEFAULT_ARTIFACT_DIR = Path("ml/artifacts")
MODEL_VERSION = "0.1.0"
ARTIFACT_STEM = "trajectory_transformer"
DISPLAY_NAME = "Trajectory Transformer"


def artifact_paths(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict[str, Path]:
    stem = f"{ARTIFACT_STEM}_v{MODEL_VERSION}"
    return {
        "network": artifact_dir / f"{stem}.pt",
        "preprocessor": artifact_dir / f"{stem}_preprocessor.joblib",
        "metadata": artifact_dir / f"{stem}_metadata.json",
    }


def load_trained_model(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> tuple[TrajectoryModel, dict] | None:
    """Returns (model, metadata), or None if the artifacts are not present --
    callers degrade to evaluating the other models rather than failing, the
    same way ml/evaluate_model.py already handles missing tree-model artifacts.
    """
    paths = artifact_paths(artifact_dir)
    if not all(path.exists() for path in paths.values()):
        return None

    with open(paths["metadata"], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = TrajectoryModel(window=metadata["window"], seed=metadata["seed"])
    model.preprocessor = joblib.load(paths["preprocessor"])

    from ml.trajectory_transformer import TrajectoryTransformer

    model.network = TrajectoryTransformer(n_features=metadata["n_features"], window=metadata["window"])
    model.network.load_state_dict(torch.load(paths["network"], weights_only=True))
    model.network.eval()
    return model, metadata


def train(
    csv_path: Path = DEFAULT_CSV_PATH,
    seed: int = DEFAULT_SEED,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    epochs: int = DEFAULT_EPOCHS,
    window: int = DEFAULT_WINDOW,
    verbose: bool = True,
) -> dict:
    split = load_and_split(csv_path)
    train_df, val_df = split["train"], split["validation"]

    model = TrajectoryModel(window=window, seed=seed)
    model.fit(train_df, epochs=epochs, verbose=verbose)

    val_probs = model.predict(val_df)
    threshold_result = select_threshold(val_df[LABEL_COLUMN].to_numpy(), val_probs)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(artifact_dir)
    n_features = model.network.input_projection.in_features

    torch.save(model.network.state_dict(), paths["network"])
    joblib.dump(model.preprocessor, paths["preprocessor"])

    metadata = {
        "model_name": DISPLAY_NAME,
        "model_version": MODEL_VERSION,
        "seed": seed,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "feature_columns": ML_FEATURE_COLUMNS,
        "n_features": n_features,
        "window": window,
        "architecture": {
            "d_model": DEFAULT_D_MODEL,
            "n_heads": DEFAULT_N_HEADS,
            "n_layers": DEFAULT_LAYERS,
            "epochs": epochs,
            "batch_size": DEFAULT_BATCH_SIZE,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "parameter_count": sum(p.numel() for p in model.network.parameters()),
        },
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
            "live case scoring, per the preference for interpretable models in anything "
            "merchant- or reviewer-facing."
        ),
    }
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"[{DISPLAY_NAME}] parameters: {metadata['architecture']['parameter_count']:,}")
        print(f"[{DISPLAY_NAME}] selected threshold: {threshold_result['selected_threshold']} ({threshold_result['selection_method']})")
        print(f"[{DISPLAY_NAME}] artifacts saved to {artifact_dir}")

    return {"model": model, "metadata": metadata, "threshold_result": threshold_result, "split": split}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the trajectory Transformer comparison baseline.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    args = parser.parse_args()
    train(Path(args.csv_path), args.seed, Path(args.artifact_dir), args.epochs, args.window)


if __name__ == "__main__":
    main()
