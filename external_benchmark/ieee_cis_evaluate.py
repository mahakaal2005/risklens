"""Held-out-test evaluation and safe aggregate-only report generation for
the IEEE-CIS external benchmark.

Persists only aggregate metrics -- no raw transaction rows, transaction
IDs, merchant IDs, identity fields, or record-level predictions are ever
written to the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from external_benchmark import ieee_cis_features, ieee_cis_loader, ieee_cis_train, ieee_cis_validate

ARTIFACT_PATH = Path("external_benchmark/artifacts/latest_ieee_cis_report.json")
FEATURE_LIST_PATH = Path("external_benchmark/artifacts/selected_feature_list.json")

NON_CLAIM_TEXT = (
    "This external benchmark uses a public anonymized transaction-level "
    "fraud dataset. It does not validate RiskLens's synthetic "
    "merchant-week refund/chargeback-loss model, merchant evidence "
    "workflow, or any Razorpay/UPI/payment-gateway integration. This "
    "benchmark is not India-specific, not Razorpay data, not UPI data, "
    "and not proof of production fraud performance."
)

FIXTURE_LABEL = "external_anonymized_transaction_fraud_benchmark_ieee_cis"


def evaluate_at_threshold(y_true: pd.Series, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "f2": float(fbeta_score(y_true, preds, beta=2.0, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc_secondary": float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else None,
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
    }


def build_report(
    split_description: dict,
    feature_list: dict,
    threshold_selection: dict,
    test_metrics: dict,
    test_prevalence: float,
) -> dict:
    """Assemble the full safe, aggregate-only report. No row-level or
    identifier-level data is included anywhere in this structure."""

    return {
        "fixture_label": FIXTURE_LABEL,
        "non_claims": NON_CLAIM_TEXT,
        "dataset_source": "IEEE-CIS Fraud Detection (Kaggle) -- manually downloaded, not verified or redistributed by this project",
        "split": split_description,
        "selected_features": feature_list,
        "model": "LogisticRegression (class_weight=balanced), scikit-learn Pipeline",
        "threshold_selection": threshold_selection,
        "test_class_prevalence": float(test_prevalence),
        "test_metrics": test_metrics,
    }


def run_full_benchmark(dataset_path: Path | str = ieee_cis_loader.EXPECTED_DATASET_PATH) -> dict:
    """End-to-end: load -> validate -> features -> chronological split ->
    train -> select threshold on validation -> evaluate on held-out test
    -> save aggregate-only report. Raises DatasetNotFoundError with the
    manual-download message if the file is missing."""

    df = ieee_cis_loader.load_train_transaction(dataset_path)
    ieee_cis_validate.validate_transaction_dataset(df)

    feature_frame = ieee_cis_features.build_feature_frame(df)
    numeric_features, categorical_features = ieee_cis_features.selected_feature_columns(feature_frame)
    feature_list = ieee_cis_features.save_feature_list(feature_frame, FEATURE_LIST_PATH)

    combined = feature_frame.copy()
    combined["TransactionDT"] = df["TransactionDT"]
    combined["isFraud"] = df["isFraud"]

    train_df, val_df, test_df = ieee_cis_train.chronological_split(combined)
    split_description = ieee_cis_train.describe_split(train_df, val_df, test_df)

    pipeline = ieee_cis_train.build_pipeline(numeric_features, categorical_features)
    x_train = train_df[numeric_features + categorical_features]
    y_train = train_df["isFraud"]
    pipeline.fit(x_train, y_train)

    x_val = val_df[numeric_features + categorical_features]
    y_val = val_df["isFraud"]
    probs_val = pipeline.predict_proba(x_val)[:, 1]
    threshold_selection = ieee_cis_train.select_threshold(y_val, probs_val)
    selected_threshold = threshold_selection["selected"]["threshold"]

    x_test = test_df[numeric_features + categorical_features]
    y_test = test_df["isFraud"]
    probs_test = pipeline.predict_proba(x_test)[:, 1]
    test_metrics = evaluate_at_threshold(y_test, probs_test, selected_threshold)
    test_prevalence = ieee_cis_validate.class_prevalence(test_df)

    report = build_report(split_description, feature_list, threshold_selection, test_metrics, test_prevalence)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {ARTIFACT_PATH}")
    return report


if __name__ == "__main__":
    run_full_benchmark()
