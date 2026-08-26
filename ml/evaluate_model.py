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
    rules_only_positive_prediction,
)
from ml.rules_engine import score_merchant_week
from ml.split_data import DEFAULT_CSV_PATH, load_and_split

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


def evaluate_split(df: pd.DataFrame, pipeline, threshold: float) -> dict:
    y_true = df[LABEL_COLUMN].values

    rules_results = _score_rules_for_split(df)
    rules_pred = np.array([1 if rules_only_positive_prediction(r) else 0 for r in rules_results])
    rules_score_surrogate = np.array([r["risk_score"] / 100.0 for r in rules_results])
    rules_metrics = compute_metrics(y_true, rules_pred, rules_score_surrogate)
    rules_metrics["pr_auc_note"] = "Computed using risk_score/100 as a monotonic severity surrogate, not a calibrated probability."
    rules_metrics["seasonal_false_positives"] = count_seasonal_false_positives(df, rules_pred)
    rules_metrics["early_hidden_false_negatives"] = count_early_hidden_false_negatives(df, rules_pred)

    X = _design_matrix(df)
    ml_probs = pipeline.predict_proba(X)[:, 1]
    ml_pred = (ml_probs >= threshold).astype(int)
    ml_metrics = compute_metrics(y_true, ml_pred, ml_probs)
    ml_metrics["seasonal_false_positives"] = count_seasonal_false_positives(df, ml_pred)
    ml_metrics["early_hidden_false_negatives"] = count_early_hidden_false_negatives(df, ml_pred)

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

    return {"rules_only": rules_metrics, "ml_only": ml_metrics, "combined": combined_metrics}


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


def evaluate(csv_path: Path = DEFAULT_CSV_PATH, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict:
    split = load_and_split(csv_path)
    metadata_path = artifact_dir / f"logistic_regression_v{MODEL_VERSION}_metadata.json"
    model_path = artifact_dir / f"logistic_regression_v{MODEL_VERSION}.joblib"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    pipeline = joblib.load(model_path)
    threshold = metadata["selected_threshold"]

    validation_results = evaluate_split(split["validation"], pipeline, threshold)
    test_results = evaluate_split(split["test"], pipeline, threshold)

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
