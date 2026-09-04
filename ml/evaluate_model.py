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

from ml.calibration import calibrate_and_evaluate
from ml.cost_model import (
    confusion_at,
    cost_metrics,
    load_cost_config,
    select_cost_optimal_threshold,
    threshold_sensitivity,
)
from ml.evaluation_report import build_report, save_report
from ml.features import LABEL_COLUMN, LATENT_STATE_COLUMN
from ml.inspect_synthetic_data import scan_leakage_column_names
from ml.model_utils import (
    check_near_perfect,
    combined_policy,
    compute_metrics,
    compute_scenario_difficulty,
    rules_only_positive_prediction,
)
from ml.model_utils import design_matrix as _design_matrix
from ml.rules_engine import score_merchant_week
from ml.train_trajectory_transformer import load_trained_model
from ml.split_data import DEFAULT_CSV_PATH, load_and_split
from ml.train_tree_models import TREE_MODELS

DEFAULT_ARTIFACT_DIR = Path("ml/artifacts")
DEFAULT_DATASET_METADATA_PATH = Path("demo_data/synthetic_data_metadata.json")
MODEL_VERSION = "0.1.0"

SEASONAL_STATE = "seasonal_sale_legitimate_returns"
EARLY_HIDDEN_STATE = "early_hidden_risk"


def _score_rules_for_split(df: pd.DataFrame) -> list[dict]:
    records = df.drop(columns=[LABEL_COLUMN, LATENT_STATE_COLUMN]).to_dict(orient="records")
    return [score_merchant_week(r) for r in records]


def count_seasonal_false_positives(df: pd.DataFrame, y_pred: np.ndarray) -> int:
    mask = (df[LATENT_STATE_COLUMN] == SEASONAL_STATE) & (df[LABEL_COLUMN] == 0) & (y_pred == 1)
    return int(mask.sum())


def count_early_hidden_false_negatives(df: pd.DataFrame, y_pred: np.ndarray) -> int:
    mask = (df[LATENT_STATE_COLUMN] == EARLY_HIDDEN_STATE) & (df[LABEL_COLUMN] == 1) & (y_pred == 0)
    return int(mask.sum())


def evaluate_split(
    df: pd.DataFrame,
    ml_pipelines: dict,
    threshold: float,
    precomputed_probs: dict | None = None,
    thresholds_by_method: dict | None = None,
) -> dict:
    """ml_pipelines maps a method name (e.g. "ml_only", "random_forest",
    "gradient_boosting") to a fitted pipeline. "ml_only" (Logistic
    Regression) is required -- it is also the model combined_policy() uses,
    per the design decision to keep Logistic Regression as the sole
    live-scoring model. Any other pipelines are evaluated the same way,
    purely as comparison baselines.

    precomputed_probs carries probabilities for methods that are not sklearn
    pipelines (the trajectory Transformer scores whole merchant sequences, not
    one design-matrix row at a time, so it cannot expose .predict_proba(X)).
    Those methods go through the identical downstream metric, scenario-difficulty,
    and threshold logic as everything else.

    thresholds_by_method lets each model be evaluated at the threshold IT
    selected on validation data. Falls back to `threshold` (Logistic
    Regression's) for anything unlisted. Applying one model's operating point
    to another model's probabilities is not a like-for-like comparison --
    each model's threshold was chosen against its own probability
    distribution.
    """
    precomputed_probs = precomputed_probs or {}
    thresholds_by_method = thresholds_by_method or {}
    y_true = df[LABEL_COLUMN].values

    rules_results = _score_rules_for_split(df)
    rules_pred = np.array([1 if rules_only_positive_prediction(r) else 0 for r in rules_results])
    rules_score_surrogate = np.array([r["risk_score"] / 100.0 for r in rules_results])
    rules_metrics = compute_metrics(y_true, rules_pred, rules_score_surrogate)
    rules_metrics["pr_auc_note"] = "Computed using risk_score/100 as a monotonic severity surrogate, not a calibrated probability."
    rules_metrics["threshold_used"] = None  # rules fire on their own conditions, not a probability cut point
    rules_metrics["seasonal_false_positives"] = count_seasonal_false_positives(df, rules_pred)
    rules_metrics["early_hidden_false_negatives"] = count_early_hidden_false_negatives(df, rules_pred)

    results = {"rules_only": rules_metrics}
    method_predictions_for_difficulty = {"rules_only": rules_pred}

    X = _design_matrix(df)
    probs_by_method = {name: pipeline.predict_proba(X)[:, 1] for name, pipeline in ml_pipelines.items()}
    probs_by_method.update(precomputed_probs)

    ml_probs_by_method = {}
    for method_name, probs in probs_by_method.items():
        method_threshold = thresholds_by_method.get(method_name, threshold)
        pred = (probs >= method_threshold).astype(int)
        metrics = compute_metrics(y_true, pred, probs)
        metrics["threshold_used"] = method_threshold
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
    combined_metrics["threshold_used"] = threshold  # combined policy is built on Logistic Regression's operating point
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
    thresholds = {}
    for model_key, spec in TREE_MODELS.items():
        model_path = artifact_dir / f"{spec['artifact_stem']}_v{spec['model_version']}.joblib"
        metadata_path = artifact_dir / f"{spec['artifact_stem']}_v{spec['model_version']}_metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            print(f"[evaluate_model] {spec['display_name']} artifact not found at {model_path} -- skipping. "
                  f"Run `python3 -m ml.train_tree_models` to generate it.")
            continue
        pipelines[model_key] = joblib.load(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            thresholds[model_key] = json.load(f)["selected_threshold"]
    return pipelines, thresholds


def _build_cost_analysis(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ml_pipelines: dict,
    threshold: float,
    thresholds_by_method: dict,
    validation_probs: dict,
    test_probs: dict,
    cost_config: dict,
    fixed_point_confusions: dict | None = None,
) -> dict:
    """Per-method rupee cost at the F2 operating point vs. the cost-optimal one.

    Both thresholds come from validation data; the test set is only scored.
    Reporting both is deliberate: the cost-optimal point is only as good as the
    assumed cost ratio, so presenting it alone would disguise a guess as a
    finding. The sensitivity table alongside it shows how much the answer
    actually depends on that guess.
    """
    val_probs_by_method = {
        name: pipeline.predict_proba(_design_matrix(val_df))[:, 1] for name, pipeline in ml_pipelines.items()
    }
    val_probs_by_method.update(validation_probs)
    test_probs_by_method = {
        name: pipeline.predict_proba(_design_matrix(test_df))[:, 1] for name, pipeline in ml_pipelines.items()
    }
    test_probs_by_method.update(test_probs)

    y_val = val_df[LABEL_COLUMN].values
    y_test = test_df[LABEL_COLUMN].values

    methods = {}
    for method_name, val_probs in val_probs_by_method.items():
        f2_threshold = thresholds_by_method.get(method_name, threshold)
        cost_result = select_cost_optimal_threshold(y_val, val_probs, cost_config)
        cost_threshold = cost_result["selected_threshold"]

        probs = test_probs_by_method[method_name]
        methods[method_name] = {
            "f2_threshold": f2_threshold,
            "cost_optimal_threshold": cost_threshold,
            # Carried through so a reader can see when the "optimum" is really
            # the edge of the search grid rather than a located minimum.
            "cost_optimal_threshold_at_grid_boundary": cost_result["threshold_at_grid_boundary"],
            "grid_boundary_note": cost_result["grid_boundary_note"],
            "refined_cost_optimal_threshold": cost_result["refined_threshold"],
            "at_refined_cost_optimal_threshold": cost_metrics(
                confusion_at(y_test, probs, cost_result["refined_threshold"]), cost_config
            ),
            "at_f2_threshold": cost_metrics(confusion_at(y_test, probs, f2_threshold), cost_config),
            "at_cost_optimal_threshold": cost_metrics(confusion_at(y_test, probs, cost_threshold), cost_config),
        }

    # Rules-only and the combined policy have no tunable probability cut point
    # -- they fire on their own conditions -- so they are costed at their single
    # fixed operating point. Omitting them would have left the cost comparison
    # missing the two policies a team would actually run today.
    for method_name, confusion in (fixed_point_confusions or {}).items():
        costed = cost_metrics(confusion, cost_config)
        methods[method_name] = {
            "f2_threshold": None,
            "cost_optimal_threshold": None,
            "at_f2_threshold": costed,
            "at_cost_optimal_threshold": costed,
            "fixed_operating_point_note": (
                "This policy has no probability threshold to tune; it fires on its own "
                "conditions, so both columns report the same single operating point."
            ),
        }

    # Sensitivity is reported for the live-scoring model, since that is the one
    # an operator would actually retune.
    sensitivity = threshold_sensitivity(y_val, val_probs_by_method["ml_only"], cost_config)

    # Pre-computed three-way headline so the dashboard renders one comparison
    # rather than re-deriving "which model is best" client-side.
    model_costs = {
        name: entry["at_refined_cost_optimal_threshold"]["expected_cost_inr"]
        for name, entry in methods.items()
        if entry.get("refined_cost_optimal_threshold") is not None
    }
    headline = None
    if model_costs and "rules_only" in methods:
        best_name = min(model_costs, key=model_costs.get)
        best_cost = model_costs[best_name]
        review_all = methods["rules_only"]["at_f2_threshold"]["review_all_cost_inr"]
        rules_cost = methods["rules_only"]["at_f2_threshold"]["expected_cost_inr"]
        headline = {
            "rules_only_cost_inr": rules_cost,
            "review_everybody_cost_inr": review_all,
            "best_model_name": best_name,
            "best_model_cost_inr": best_cost,
            "best_model_margin_over_review_everybody_pct": round((review_all - best_cost) / review_all * 100, 1),
            "rules_only_beats_review_everybody": bool(rules_cost < review_all),
            "margin_note": (
                "The margin over reviewing everybody is the honest headline: most of the value at "
                "the assumed cost ratio comes from reviewing a lot, not from the model's "
                "discrimination. Read the bars against each other, not against zero."
            ),
        }

    return {
        "cost_assumptions_inr": cost_config["costs_inr"],
        "recovery_rate": cost_config["recovery_rate"],
        "headline_comparison": headline,
        "assumption_notice": (
            "Every rupee figure below is an unverified assumption from rules/cost_model.yaml, "
            "chosen to make the cost tradeoff visible on synthetic data. They are not Razorpay's "
            "costs, not any real provider's costs, and not benchmarked against published figures. "
            "Replace them with measured costs before drawing any operational conclusion."
        ),
        "methods": methods,
        "threshold_sensitivity_logistic_regression": sensitivity,
    }


def _build_calibration_analysis(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ml_pipelines: dict,
    validation_probs: dict,
    test_probs: dict,
) -> dict:
    """Per-method calibration: is a stated 0.86 actually an 86% chance?

    Only probability-producing methods appear. Rules-only has no calibrated
    probability (its PR-AUC already uses risk_score/100 as an explicitly
    uncalibrated severity surrogate), and the combined policy emits a
    recommendation rather than a probability, so calibrating either would be
    measuring something that does not exist.
    """
    val_by_method = {
        name: pipeline.predict_proba(_design_matrix(val_df))[:, 1] for name, pipeline in ml_pipelines.items()
    }
    val_by_method.update(validation_probs)
    test_by_method = {
        name: pipeline.predict_proba(_design_matrix(test_df))[:, 1] for name, pipeline in ml_pipelines.items()
    }
    test_by_method.update(test_probs)

    y_val = val_df[LABEL_COLUMN].values
    y_test = test_df[LABEL_COLUMN].values

    methods = {}
    for method_name, val_probs in val_by_method.items():
        result = calibrate_and_evaluate(y_val, val_probs, y_test, test_by_method[method_name])
        # The calibrated probability arrays are working data, not report
        # payload -- 14,400 floats per variant per method would bloat an API
        # response for no reader benefit.
        methods[method_name] = {
            "metrics": result["metrics"],
            "best_by_brier_score": result["best_by_brier_score"],
        }

    return {
        "fit_on": "validation_split",
        "scored_on": "held_out_test_split",
        "methodology_note": (
            "Calibrators are fit on the validation split, which this project also uses for "
            "threshold selection -- validation does double duty. The held-out test set is "
            "scored only, never fit on, so these test metrics are honest; but a calibrator fit "
            "on genuinely fresh data would be a stricter test."
        ),
        "interpretation_note": (
            "Brier score is a proper scoring rule and is the primary number. Expected "
            "calibration error isolates the calibration gap but is bin-count sensitive; "
            "maximum calibration error is reported because ECE's count weighting hides errors "
            "in the sparse high-probability bins that produce escalations."
        ),
        "methods": methods,
    }


def evaluate(csv_path: Path = DEFAULT_CSV_PATH, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict:
    split = load_and_split(csv_path)
    metadata_path = artifact_dir / f"logistic_regression_v{MODEL_VERSION}_metadata.json"
    model_path = artifact_dir / f"logistic_regression_v{MODEL_VERSION}.joblib"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    lr_pipeline = joblib.load(model_path)
    threshold = metadata["selected_threshold"]

    tree_pipelines, tree_thresholds = _load_tree_model_pipelines(artifact_dir)
    ml_pipelines = {"ml_only": lr_pipeline, **tree_pipelines}
    thresholds_by_method = {"ml_only": threshold, **tree_thresholds}

    trajectory = load_trained_model(artifact_dir)
    validation_probs = {}
    test_probs = {}
    if trajectory is None:
        print("[evaluate_model] Trajectory Transformer artifact not found -- skipping. "
              "Run `python3 -m ml.train_trajectory_transformer` to generate it.")
    else:
        trajectory_model, trajectory_metadata = trajectory
        # The trailing window must not be truncated at the split boundary: a
        # merchant's first test weeks have real prior history in the validation
        # split. Feature vectors only, weeks strictly earlier only -- see
        # build_sequences(). The sklearn models are per-row and need no
        # equivalent, so their path is untouched.
        full_history = pd.concat([split["train"], split["validation"], split["test"]], ignore_index=True)
        validation_probs["trajectory_transformer"] = trajectory_model.predict(
            split["validation"], history_df=full_history
        )
        test_probs["trajectory_transformer"] = trajectory_model.predict(
            split["test"], history_df=full_history
        )
        thresholds_by_method["trajectory_transformer"] = trajectory_metadata["selected_threshold"]

    validation_results, _ = evaluate_split(
        split["validation"], ml_pipelines, threshold,
        precomputed_probs=validation_probs, thresholds_by_method=thresholds_by_method,
    )
    test_results, scenario_difficulty = evaluate_split(
        split["test"], ml_pipelines, threshold,
        precomputed_probs=test_probs, thresholds_by_method=thresholds_by_method,
    )

    # Rupee cost analysis. Threshold selection stays on validation data only,
    # exactly like the F2 threshold; the test set is only ever *scored* at an
    # already-fixed operating point, never used to choose one.
    cost_config = load_cost_config()
    cost_analysis = _build_cost_analysis(
        split["validation"], split["test"], ml_pipelines, threshold,
        thresholds_by_method, validation_probs, test_probs, cost_config,
        fixed_point_confusions={
            name: test_results[name]["confusion_matrix"]
            for name in ("rules_only", "combined")
            if name in test_results
        },
    )

    calibration = _build_calibration_analysis(
        split["validation"], split["test"], ml_pipelines, validation_probs, test_probs,
    )

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
        "cost_analysis": cost_analysis,
        "calibration": calibration,
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

    cost = result["cost_analysis"]
    print("\nRupee cost analysis (held-out test) -- ALL FIGURES ASSUMED, NOT MEASURED:")
    print(f"  assumptions: {cost['cost_assumptions_inr']}, recovery_rate={cost['recovery_rate']}")
    for name, entry in cost["methods"].items():
        at_f2 = entry["at_f2_threshold"]
        at_cost = entry["at_cost_optimal_threshold"]
        if entry["f2_threshold"] is None:
            print(f"  {name}: fixed operating point (no tunable threshold) -- "
                  f"cost=Rs{at_f2['expected_cost_inr']:,.0f} reviews={at_f2['n_reviews_generated']} "
                  f"savings_per_1000_reviews=Rs{at_f2['savings_per_1000_reviews_inr']:,.0f} "
                  f"beats_review_all={at_f2['beats_review_all']}")
            continue
        boundary = " [AT GRID EDGE -- grid too coarse to locate the optimum]" if entry["cost_optimal_threshold_at_grid_boundary"] else ""
        print(f"  {name}: f2_threshold={entry['f2_threshold']} cost_optimal_threshold={entry['cost_optimal_threshold']}{boundary}")
        print(f"    at F2 threshold:   cost=Rs{at_f2['expected_cost_inr']:,.0f} reviews={at_f2['n_reviews_generated']} "
              f"savings_per_1000_reviews=Rs{at_f2['savings_per_1000_reviews_inr']:,.0f} beats_review_all={at_f2['beats_review_all']}")
        print(f"    at cost-optimal:   cost=Rs{at_cost['expected_cost_inr']:,.0f} reviews={at_cost['n_reviews_generated']} "
              f"savings_per_1000_reviews=Rs{at_cost['savings_per_1000_reviews_inr']:,.0f} beats_review_all={at_cost['beats_review_all']}")

    print("\nThreshold sensitivity to the cost ratio (Logistic Regression, validation-selected):")
    for row in cost["threshold_sensitivity_logistic_regression"]:
        print(f"  missed_loss/review_cost={row['missed_loss_to_review_cost_ratio']:>4} "
              f"(Rs{row['implied_missed_loss_inr']:,}) -> threshold={row['cost_optimal_threshold']} "
              f"reviews={row['n_reviews_generated']} savings_per_1000_reviews=Rs{row['savings_per_1000_reviews_inr']:,.0f}")

    print("\nCalibration (fit on validation, scored on held-out test):")
    for name, entry in result["calibration"]["methods"].items():
        raw = entry["metrics"]["raw"]
        print(f"  {name}: best_by_brier={entry['best_by_brier_score']} "
              f"(raw brier={raw['brier_score']:.4f} ece={raw['expected_calibration_error']:.4f} "
              f"mce={raw['maximum_calibration_error']:.4f} mean_pred={raw['mean_predicted_probability']:.4f} "
              f"observed={raw['observed_positive_rate']:.4f})")
        for variant in ("platt", "isotonic"):
            m = entry["metrics"][variant]
            print(f"      {variant:9s} brier={m['brier_score']:.4f} ece={m['expected_calibration_error']:.4f} "
                  f"mce={m['maximum_calibration_error']:.4f}")

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
