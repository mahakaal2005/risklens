"""Milestone 3: evaluate the rules-only baseline, the Logistic Regression
model, and the combined model+rules policy on validation and held-out test
data, and run the near-perfect-score investigation gate if triggered.

This script never selects or adjusts anything using the held-out test set --
the threshold was already fixed by train_baseline_model.py using validation
data only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.evaluation_report import build_report, save_report
from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN, compute_feature_frame
from ml.inspect_synthetic_data import scan_leakage_column_names
from ml.model_utils import (
    ML_FEATURE_COLUMNS,
    check_near_perfect,
    combined_policy,
    compute_metrics,
    compute_scenario_difficulty,
    rules_only_positive_prediction,
)
from ml.rules_engine import score_merchant_week
from ml.split_data import DEFAULT_CSV_PATH, load_and_split
from ml.train_tree_models import TREE_MODELS

DEFAULT_ARTIFACT_DIR = Path("ml/artifacts")
DEFAULT_DATASET_METADATA_PATH = Path("demo_data/synthetic_data_metadata.json")
MODEL_VERSION = "0.1.0"

SEASONAL_STATE = "seasonal_sale_legitimate_returns"
EARLY_HIDDEN_STATE = "early_hidden_risk"


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return compute_feature_frame(df)[ML_FEATURE_COLUMNS]


def _score_rules_for_split(df: pd.DataFrame) -> list[dict]:
    records = df.drop(columns=[LABEL_COLUMN, LATENT_STATE_COLUMN]).to_dict(orient="records")
    return [score_merchant_week(r) for r in records]


def count_seasonal_false_positives(df: pd.DataFrame, y_pred: np.ndarray) -> int:
    mask = (df[LATENT_STATE_COLUMN] == SEASONAL_STATE) & (df[LABEL_COLUMN] == 0) & (y_pred == 1)
    return int(mask.sum())


def count_early_hidden_false_negatives(df: pd.DataFrame, y_pred: np.ndarray) -> int:
    mask = (df[LATENT_STATE_COLUMN] == EARLY_HIDDEN_STATE) & (df[LABEL_COLUMN] == 1) & (y_pred == 0)
    return int(mask.sum())


def evaluate_split(df: pd.DataFrame, ml_pipelines: dict, threshold: float) -> dict:
    """ml_pipelines maps a method name (e.g. "ml_only", "random_forest",
    "gradient_boosting") to a fitted pipeline. "ml_only" (Logistic
    Regression) is required -- it is also the model combined_policy() uses,
    per the design decision to keep Logistic Regression as the sole
    live-scoring model. Any other pipelines are evaluated the same way,
    purely as comparison baselines.
    """
    y_true = df[LABEL_COLUMN].values

    rules_results = _score_rules_for_split(df)
    rules_pred = np.array([1 if rules_only_positive_prediction(r) else 0 for r in rules_results])
    rules_score_surrogate = np.array([r["risk_score"] / 100.0 for r in rules_results])
    rules_metrics = compute_metrics(y_true, rules_pred, rules_score_surrogate)
    rules_metrics["pr_auc_note"] = "Computed using risk_score/100 as a monotonic severity surrogate, not a calibrated probability."
    rules_metrics["seasonal_false_positives"] = count_seasonal_false_positives(df, rules_pred)
    rules_metrics["early_hidden_false_negatives"] = count_early_hidden_false_negatives(df, rules_pred)

    results = {"rules_only": rules_metrics}
    method_predictions_for_difficulty = {"rules_only": rules_pred}

    X = _design_matrix(df)
    ml_probs_by_method = {}
    for method_name, pipeline in ml_pipelines.items():
        probs = pipeline.predict_proba(X)[:, 1]
        pred = (probs >= threshold).astype(int)
        metrics = compute_metrics(y_true, pred, probs)
        metrics["seasonal_false_positives"] = count_seasonal_false_positives(df, pred)
        metrics["early_hidden_false_negatives"] = count_early_hidden_false_negatives(df, pred)
        results[method_name] = metrics
        method_predictions_for_difficulty[method_name] = pred
        ml_probs_by_method[method_name] = probs

    ml_probs = ml_probs_by_method["ml_only"]
    combined_decisions = [
        combined_policy(ml_probs[i], threshold, set(rules_results[i]["triggered_rules"]))
        for i in range(len(df))
    ]
    combined_pred = np.array([
        1 if d.recommendation in {"REQUEST_EVIDENCE", "MANUAL_REVIEW_REQUIRED", "ESCALATE_TO_COMPLIANCE"} else 0
        for d in combined_decisions
    ])
    combined_metrics = compute_metrics(y_true, combined_pred, ml_probs)
    combined_metrics["seasonal_false_positives"] = count_seasonal_false_positives(df, combined_pred)
    combined_metrics["early_hidden_false_negatives"] = count_early_hidden_false_negatives(df, combined_pred)
    combined_metrics["recommendation_distribution"] = pd.Series([d.recommendation for d in combined_decisions]).value_counts().to_dict()
    results["combined"] = combined_metrics
    method_predictions_for_difficulty["combined"] = combined_pred

    scenario_difficulty = compute_scenario_difficulty(df, LATENT_STATE_COLUMN, LABEL_COLUMN, method_predictions_for_difficulty)

    return results, scenario_difficulty


def run_investigation(df: pd.DataFrame, split_info: dict, feature_columns: list[str]) -> dict:
    findings = {}

    leakage_scan = scan_leakage_column_names(feature_columns)
    findings["label_leakage_check"] = {
        "candidate_features_scanned": feature_columns,
        "unexplained_matches": leakage_scan["unexplained_matches"],
        "passed": len(leakage_scan["unexplained_matches"]) == 0,
    }

    train_max = split_info["train"]["week_start_max"]
    val_min = split_info["validation"]["week_start_min"]
    val_max = split_info["validation"]["week_start_max"]
    test_min = split_info["test"]["week_start_min"]
    time_ordered = train_max < val_min <= val_max < test_min
    findings["time_leakage_check"] = {"train_max": train_max, "validation_range": [val_min, val_max], "test_min": test_min, "passed": bool(time_ordered)}

    findings["split_overlap_check"] = {"passed": bool(time_ordered), "note": "Derived from chronological ordering above; ml/split_data.py assigns each week to exactly one split by construction."}

    duplicate_count = int(df.duplicated(subset=["merchant_id", "week_start"]).sum())
    findings["duplicated_entity_row_check"] = {"duplicate_count": duplicate_count, "passed": duplicate_count == 0}

    findings["feature_list_inspection"] = {"features": feature_columns, "passed": LATENT_STATE_COLUMN not in feature_columns and LABEL_COLUMN not in feature_columns}
    findings["latent_state_exclusion_check"] = {"passed": LATENT_STATE_COLUMN not in feature_columns}

    seasonal_present = (df[LATENT_STATE_COLUMN] == SEASONAL_STATE).any()
    early_hidden_present = (df[LATENT_STATE_COLUMN] == EARLY_HIDDEN_STATE).any()
    findings["seasonal_and_early_hidden_presence_check"] = {"seasonal_sale_rows_present": bool(seasonal_present), "early_hidden_risk_rows_present": bool(early_hidden_present)}

    return findings


def _load_tree_model_pipelines(artifact_dir: Path) -> dict:
    """Loads Random Forest / Gradient Boosting artifacts if present.
    Missing artifacts degrade gracefully (a clear printed message, LR-only
    evaluation continues) rather than raising -- consistent with the
    project's "fail safely" principle; these are optional comparison
    models, not part of the required scoring path.
    """
    pipelines = {}
    for model_key, spec in TREE_MODELS.items():
        model_path = artifact_dir / f"{spec['artifact_stem']}_v{spec['model_version']}.joblib"
        if not model_path.exists():
            print(f"[evaluate_model] {spec['display_name']} artifact not found at {model_path} -- skipping. "
                  f"Run `python3 -m ml.train_tree_models` to generate it.")
            continue
        pipelines[model_key] = joblib.load(model_path)
    return pipelines


def evaluate(csv_path: Path = DEFAULT_CSV_PATH, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict:
    split = load_and_split(csv_path)
    metadata_path = artifact_dir / f"logistic_regression_v{MODEL_VERSION}_metadata.json"
    model_path = artifact_dir / f"logistic_regression_v{MODEL_VERSION}.joblib"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    lr_pipeline = joblib.load(model_path)
    threshold = metadata["selected_threshold"]

    ml_pipelines = {"ml_only": lr_pipeline, **_load_tree_model_pipelines(artifact_dir)}

    validation_results, _ = evaluate_split(split["validation"], ml_pipelines, threshold)
    test_results, scenario_difficulty = evaluate_split(split["test"], ml_pipelines, threshold)

    gate_outcomes = {}
    investigation_triggered = False
    for method_name, metrics in test_results.items():
        triggered, reason = check_near_perfect(metrics)
        gate_outcomes[method_name] = {"under_investigation": triggered, "reason": reason}
        if triggered:
            investigation_triggered = True

    investigation = None
    if investigation_triggered:
        investigation = run_investigation(split["test"], split["split_info"], metadata["feature_columns"])

    return {
        "threshold": threshold,
        "split_info": split["split_info"],
        "validation_results": validation_results,
        "test_results": test_results,
        "scenario_difficulty": scenario_difficulty,
        "gate_outcomes": gate_outcomes,
        "investigation_triggered": investigation_triggered,
        "investigation": investigation,
    }


def _print_metrics(label: str, metrics: dict) -> None:
    print(f"  {label}: prevalence={metrics['prevalence']} precision={metrics['precision']} recall={metrics['recall']} "
          f"f1={metrics['f1']} f2={metrics['f2']} pr_auc={metrics['pr_auc']} roc_auc(secondary)={metrics['roc_auc_secondary']} "
          f"fpr={metrics['false_positive_rate']} fnr={metrics['false_negative_rate']} "
          f"n_predicted_positive={metrics['n_predicted_positive']} cm={metrics['confusion_matrix']} "
          f"seasonal_fp={metrics['seasonal_false_positives']} early_hidden_fn={metrics['early_hidden_false_negatives']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Milestone 3 rules-only, ML-only, and combined policies.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--dataset-metadata-path", type=str, default=str(DEFAULT_DATASET_METADATA_PATH))
    args = parser.parse_args()

    result = evaluate(Path(args.csv_path), Path(args.artifact_dir))

    print(f"Selected operating threshold: {result['threshold']}")
    print("\nValidation results:")
    for name, metrics in result["validation_results"].items():
        _print_metrics(name, metrics)

    print("\nHeld-out test results:")
    for name, metrics in result["test_results"].items():
        _print_metrics(name, metrics)

    print("\nScenario difficulty (held-out test, by latent state):")
    for entry in result["scenario_difficulty"]:
        print(f"  {entry['state']}: rows={entry['row_count']} positive_rate={entry['positive_rate']}")
        for method_name, breakdown in entry["methods"].items():
            print(f"    {method_name}: recall_within_state={breakdown['recall_within_state']} predicted_positive_rate={breakdown['predicted_positive_rate']}")

    print("\nNear-perfect-score gate outcomes (held-out test):")
    for name, outcome in result["gate_outcomes"].items():
        status = "UNDER INVESTIGATION" if outcome["under_investigation"] else "APPROVED"
        print(f"  {name}: {status} -- {outcome['reason']}")

    if result["investigation_triggered"]:
        print("\nInvestigation findings:")
        print(json.dumps(result["investigation"], indent=2, default=str))

    artifact_dir = Path(args.artifact_dir)
    model_metadata_path = artifact_dir / f"logistic_regression_v{MODEL_VERSION}_metadata.json"
    with open(model_metadata_path, "r", encoding="utf-8") as f:
        model_metadata = json.load(f)

    dataset_metadata_path = Path(args.dataset_metadata_path)
    dataset_metadata = {}
    if dataset_metadata_path.exists():
        with open(dataset_metadata_path, "r", encoding="utf-8") as f:
            dataset_metadata = json.load(f)

    report = build_report(result, dataset_metadata, model_metadata)
    written_paths = save_report(report, path=artifact_dir / "latest_evaluation_report.json")
    print(f"\nEvaluation report written to: {[str(p) for p in written_paths]}")


if __name__ == "__main__":
    main()
