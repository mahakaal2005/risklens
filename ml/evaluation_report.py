"""Milestone 7: build, save, load, and validate the persisted evaluation
report that GET /metrics reads from -- so the API never retrains or
re-scores on request.

The report is an aggregate-only artifact: it contains held-out-test
metrics (precision/recall/PR-AUC/confusion-matrix/etc.) for the three
comparison methods, plus dataset/split/model metadata. It never contains
a per-row label, a per-row latent state, a merchant ID, a raw model
coefficient, or any enforcement instruction.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

REPORT_VERSION = "1.0"
DEFAULT_REPORT_PATH = Path("ml/artifacts/latest_evaluation_report.json")
DATA_MODE = "synthetic-only"

SYNTHETIC_DATA_NOTICE = (
    "Synthetic-data metrics demonstrate prototype workflow only and do not prove "
    "real-world chargeback-risk performance."
)

LIMITATIONS = [
    "Synthetic data only.",
    "Metrics demonstrate the prototype workflow and do not prove real-world chargeback-risk performance.",
    "No real payment gateway, merchant, settlement, enforcement, or fraud decision data is used.",
]

NEAR_PERFECT_CONDITIONS_CHECKED = [
    "PR-AUC >= 0.98",
    "precision >= 0.98 and recall >= 0.98",
    "zero false positives or zero false negatives",
]

EXCLUDED_MODEL_FIELDS = ["merchant_id", "week_start", "label_high_loss_next_30d", "latent_state_for_demo_only"]

# Maps the internal latent-state token to a safe, generic scenario label for
# the report -- mirrors the existing pattern of "seasonal_false_positives" /
# "early_hidden_false_negatives" (a generic name tied to a state's role,
# not the raw internal token) so the new scenario_difficulty section
# doesn't introduce a new way for a prohibited raw state string to appear
# in an aggregate, analyst-facing report.
SCENARIO_LABEL_MAP = {
    "stable_merchant": "stable",
    "seasonal_sale_legitimate_returns": "seasonal_returns",
    "operational_fulfilment_failure": "operational_failure",
    "high_risk_merchant_behaviour": "high_risk_behavior",
    "early_hidden_risk": "early_hidden_risk",
}

_METHOD_KEY_RENAME = {
    "rules_only": "rules_only",
    "ml_only": "logistic_regression",
    "random_forest": "random_forest",
    "gradient_boosting": "gradient_boosting",
    "trajectory_transformer": "trajectory_transformer",
    "combined": "combined_policy",
}

REQUIRED_TOP_LEVEL_KEYS = {
    "report_version", "generated_at", "data_mode", "synthetic_data_notice",
    "dataset", "split", "model", "methods", "scenario_difficulty",
    "near_perfect_score_investigation", "limitations",
}

# Analyses added after the report format was first fixed. Permitted and
# strictly validated when present, but a report generated before they existed
# must still validate -- otherwise GET /metrics would reject the whole report
# rather than serving null for the missing section, which is the contract the
# dashboard's placeholder rendering depends on.
OPTIONAL_TOP_LEVEL_KEYS = {"cost_analysis", "calibration"}
# random_forest / gradient_boosting are comparison-only baselines (see
# ml/train_tree_models.py) -- required whenever present in the evaluate()
# result, but evaluate() degrades gracefully if their artifacts are
# missing, so they are validated here only when the report actually
# contains them (see the methods-required check below).
CORE_REQUIRED_METHOD_NAMES = {"rules_only", "logistic_regression", "combined_policy"}
OPTIONAL_METHOD_NAMES = {"random_forest", "gradient_boosting", "trajectory_transformer"}
REQUIRED_METHOD_NAMES = CORE_REQUIRED_METHOD_NAMES | OPTIONAL_METHOD_NAMES
REQUIRED_METHOD_METRIC_KEYS = {
    "precision", "recall", "f1", "f2", "pr_auc", "roc_auc",
    "false_positive_rate", "false_negative_rate", "confusion_matrix",
    "predicted_review_case_count", "seasonal_sale_false_positive_count",
    "early_hidden_risk_false_negative_count", "threshold_used",
}
# threshold_used is included so the comparison table can show that each model
# was evaluated at the operating point IT selected on validation data, rather
# than at another model's operating point.
RATE_FIELDS = {"precision", "recall", "f1", "f2", "pr_auc", "roc_auc", "false_positive_rate", "false_negative_rate", "threshold_used"}

# Prohibited anywhere in the serialized report EXCEPT inside
# model.excluded_fields, which is required by the report schema to
# literally name label_high_loss_next_30d and latent_state_for_demo_only
# as documentation of what was excluded from training -- that is a safe,
# intentional use of the column NAME, not a leak of a per-row VALUE.
FIELD_NAME_DOCUMENTATION_ONLY = {"label_high_loss_next_30d", "latent_state_for_demo_only"}

PROHIBITED_LITERAL_STRINGS = [
    "stable_merchant", "seasonal_sale_legitimate_returns", "operational_fulfilment_failure",
    "high_risk_merchant_behaviour", "early_hidden_risk_case",
    "freeze", "ban", "terminate", "hold settlement", "reject payment",
]
MERCHANT_ID_PATTERN = re.compile(r"merchant_demo_\d+")


def _method_metrics(raw: dict) -> dict:
    return {
        "precision": raw["precision"],
        "recall": raw["recall"],
        "f1": raw["f1"],
        "f2": raw["f2"],
        "pr_auc": raw["pr_auc"],
        "roc_auc": raw.get("roc_auc_secondary", raw.get("roc_auc")),
        "false_positive_rate": raw["false_positive_rate"],
        "false_negative_rate": raw["false_negative_rate"],
        "confusion_matrix": raw["confusion_matrix"],
        "predicted_review_case_count": raw.get("n_predicted_positive", raw.get("predicted_review_case_count")),
        "seasonal_sale_false_positive_count": raw.get("seasonal_false_positives", raw.get("seasonal_sale_false_positive_count")),
        "early_hidden_risk_false_negative_count": raw.get("early_hidden_false_negatives", raw.get("early_hidden_risk_false_negative_count")),
        "threshold_used": raw.get("threshold_used"),
    }


COST_ANALYSIS_KEYS = {
    "cost_assumptions_inr", "recovery_rate", "assumption_notice", "headline_comparison",
    "methods", "threshold_sensitivity_logistic_regression",
}
COST_METHOD_KEYS = {"f2_threshold", "cost_optimal_threshold", "at_f2_threshold", "at_cost_optimal_threshold"}


def _validate_cost_analysis(cost_analysis) -> list[str]:
    """The cost section is optional (an older report predates it), but when
    present it must have the exact expected shape and must carry its
    assumption notice -- a rupee figure served without the "this is a guess"
    disclaimer is the single most misreadable thing this report can emit."""
    if not cost_analysis:
        return []

    issues: list[str] = []
    if not isinstance(cost_analysis, dict):
        return ["cost_analysis must be an object"]

    unexpected = set(cost_analysis.keys()) - COST_ANALYSIS_KEYS
    if unexpected:
        issues.append(f"cost_analysis contains unexpected field(s): {sorted(unexpected)}")

    if not cost_analysis.get("assumption_notice"):
        issues.append("cost_analysis.assumption_notice must be a non-empty string")

    for method_name, entry in (cost_analysis.get("methods") or {}).items():
        if not isinstance(entry, dict) or COST_METHOD_KEYS - set(entry.keys()):
            issues.append(f"cost_analysis.methods.{method_name} is missing required keys")
            continue
        for point in ("at_f2_threshold", "at_cost_optimal_threshold"):
            savings = entry[point].get("savings_per_1000_reviews_inr")
            if savings is not None and not isinstance(savings, (int, float)):
                issues.append(f"cost_analysis.methods.{method_name}.{point}.savings_per_1000_reviews_inr must be a number or null")

    return issues


def _cost_analysis_section(cost_analysis: dict | None) -> dict:
    """Rupee cost summary, with method keys renamed to the report's public
    names so the cost section and the metrics section agree.

    The per-threshold `candidates` grids from select_cost_optimal_threshold()
    are deliberately NOT carried into the report: they are working data, and
    the report is served over an API where a smaller, fixed-shape payload is
    the safer default.
    """
    if not cost_analysis:
        return {}

    methods = {
        _METHOD_KEY_RENAME.get(name, name): entry
        for name, entry in cost_analysis.get("methods", {}).items()
    }
    return {
        "cost_assumptions_inr": cost_analysis["cost_assumptions_inr"],
        "recovery_rate": cost_analysis["recovery_rate"],
        "assumption_notice": cost_analysis["assumption_notice"],
        "headline_comparison": cost_analysis.get("headline_comparison"),
        "methods": methods,
        "threshold_sensitivity_logistic_regression": cost_analysis["threshold_sensitivity_logistic_regression"],
    }


CALIBRATION_KEYS = {"fit_on", "scored_on", "methodology_note", "interpretation_note", "methods"}
CALIBRATION_VARIANTS = {"raw", "platt", "isotonic"}
CALIBRATION_METRIC_KEYS = {
    "brier_score", "expected_calibration_error", "maximum_calibration_error",
    "mean_predicted_probability", "observed_positive_rate", "reliability_curve",
}


def _validate_calibration(calibration) -> list[str]:
    """Optional section (older reports predate it), but strictly shaped when
    present. The methodology note is required for the same reason the cost
    section's assumption notice is: a calibration number served without the
    "validation does double duty" caveat overstates how clean the measurement is.
    """
    if not calibration:
        return []
    if not isinstance(calibration, dict):
        return ["calibration must be an object"]

    issues: list[str] = []
    unexpected = set(calibration.keys()) - CALIBRATION_KEYS
    if unexpected:
        issues.append(f"calibration contains unexpected field(s): {sorted(unexpected)}")
    if not calibration.get("methodology_note"):
        issues.append("calibration.methodology_note must be a non-empty string")

    for method_name, entry in (calibration.get("methods") or {}).items():
        metrics = (entry or {}).get("metrics") or {}
        missing_variants = CALIBRATION_VARIANTS - set(metrics.keys())
        if missing_variants:
            issues.append(f"calibration.methods.{method_name} is missing variants: {sorted(missing_variants)}")
            continue
        for variant, values in metrics.items():
            if CALIBRATION_METRIC_KEYS - set(values.keys()):
                issues.append(f"calibration.methods.{method_name}.{variant} is missing required metric keys")
                continue
            for field in ("brier_score", "expected_calibration_error", "maximum_calibration_error"):
                value = values[field]
                if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                    issues.append(f"calibration.methods.{method_name}.{variant}.{field} must be a number in [0, 1], got {value!r}")

    return issues


def _calibration_section(calibration: dict | None) -> dict:
    if not calibration:
        return {}
    return {
        **{k: v for k, v in calibration.items() if k != "methods"},
        "methods": {
            _METHOD_KEY_RENAME.get(name, name): entry
            for name, entry in calibration.get("methods", {}).items()
        },
    }


def build_report(evaluate_result: dict, dataset_metadata: dict, model_metadata: dict) -> dict:
    """evaluate_result is ml.evaluate_model.evaluate()'s return value.
    dataset_metadata is demo_data/synthetic_data_metadata.json's contents.
    model_metadata is the trained model's saved metadata JSON contents.
    """
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

    split_info = evaluate_result["split_info"]
    split_section = {
        split_name: {
            "week_count": info["week_count"],
            "row_count": info["row_count"],
            "date_start": info["week_start_min"],
            "date_end": info["week_start_max"],
        }
        for split_name, info in split_info.items()
    }

    methods_section = {
        _METHOD_KEY_RENAME[name]: _method_metrics(metrics)
        for name, metrics in evaluate_result["test_results"].items()
    }

    scenario_difficulty_section = [
        {**entry, "state": SCENARIO_LABEL_MAP.get(entry["state"], entry["state"])}
        for entry in evaluate_result.get("scenario_difficulty", [])
    ]

    if evaluate_result["investigation_triggered"]:
        triggered_methods = [name for name, outcome in evaluate_result["gate_outcomes"].items() if outcome["under_investigation"]]
        investigation_section = {
            "status": "under_investigation",
            "conditions_checked": NEAR_PERFECT_CONDITIONS_CHECKED,
            "notes": f"Near-perfect gate triggered for: {', '.join(triggered_methods)}. See docs/MILESTONE_3_MODEL_EVALUATION.md for the full investigation checklist result.",
        }
    else:
        investigation_section = {
            "status": "not_triggered",
            "conditions_checked": NEAR_PERFECT_CONDITIONS_CHECKED,
            "notes": "No near-perfect gate condition was triggered.",
        }

    return {
        "report_version": REPORT_VERSION,
        "generated_at": generated_at,
        "data_mode": DATA_MODE,
        "synthetic_data_notice": SYNTHETIC_DATA_NOTICE,
        "dataset": {
            "seed": dataset_metadata.get("seed"),
            "generator_version": dataset_metadata.get("generator_version"),
            "row_count": dataset_metadata.get("row_count"),
            "merchant_count": dataset_metadata.get("merchant_count"),
            "date_range": {
                "start": dataset_metadata.get("date_range", {}).get("week_start_min"),
                "end": dataset_metadata.get("date_range", {}).get("week_start_max"),
            },
        },
        "split": split_section,
        "scenario_difficulty": scenario_difficulty_section,
        "cost_analysis": _cost_analysis_section(evaluate_result.get("cost_analysis")),
        "calibration": _calibration_section(evaluate_result.get("calibration")),
        "model": {
            "name": "Logistic Regression",
            "version": model_metadata.get("model_version"),
            "selected_threshold": evaluate_result["threshold"],
            "threshold_selection_method": model_metadata.get(
                "threshold_selection_method", "validation F2 maximization with documented precision/FPR constraints"
            ),
            "feature_count": len(model_metadata.get("feature_columns", [])),
            "excluded_fields": EXCLUDED_MODEL_FIELDS,
        },
        "methods": methods_section,
        "near_perfect_score_investigation": investigation_section,
        "limitations": LIMITATIONS,
    }


def _write_json_atomically(payload: dict, path: Path) -> None:
    """Writes to a temp file in the same directory, then atomically renames
    it into place (os.replace is atomic on POSIX and Windows), so a reader
    (e.g. GET /metrics) can never observe a partially-written/truncated
    file, even if this process is interrupted mid-write."""
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def save_report(report: dict, path: Path = DEFAULT_REPORT_PATH, also_timestamped: bool = True) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []

    _write_json_atomically(report, path)
    written.append(path)

    if also_timestamped:
        stamp = dt.datetime.fromisoformat(report["generated_at"]).strftime("%Y%m%dT%H%M%SZ")
        timestamped_path = path.parent / f"evaluation_report_{stamp}.json"
        _write_json_atomically(report, timestamped_path)
        written.append(timestamped_path)

    return written


class ReportNotFoundError(Exception):
    pass


class ReportInvalidError(Exception):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Evaluation report failed validation:\n- " + "\n- ".join(issues))


def load_report(path: Path = DEFAULT_REPORT_PATH) -> dict:
    if not path.exists():
        raise ReportNotFoundError(f"No evaluation report found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # may raise json.JSONDecodeError for corrupt files


def validate_report(report: dict) -> list[str]:
    """Returns a list of validation issues; empty list means valid."""
    issues: list[str] = []

    missing_top_level = REQUIRED_TOP_LEVEL_KEYS - set(report.keys())
    if missing_top_level:
        issues.append(f"Missing required top-level fields: {sorted(missing_top_level)}")
        return issues  # cannot safely check further without these

    unexpected_top_level = set(report.keys()) - REQUIRED_TOP_LEVEL_KEYS - OPTIONAL_TOP_LEVEL_KEYS
    if unexpected_top_level:
        issues.append(f"Report contains unexpected top-level field(s), which is rejected as a safety precaution: {sorted(unexpected_top_level)}")

    if report.get("data_mode") != DATA_MODE:
        issues.append(f"data_mode must equal '{DATA_MODE}'")

    if not report.get("synthetic_data_notice"):
        issues.append("synthetic_data_notice must be a non-empty string")

    if not report.get("limitations"):
        issues.append("limitations must be a non-empty list")

    issues.extend(_validate_cost_analysis(report.get("cost_analysis")))
    issues.extend(_validate_calibration(report.get("calibration")))

    methods = report.get("methods", {})
    missing_methods = CORE_REQUIRED_METHOD_NAMES - set(methods.keys())
    if missing_methods:
        issues.append(f"methods is missing required entries: {sorted(missing_methods)}")

    unknown_methods = set(methods.keys()) - REQUIRED_METHOD_NAMES
    if unknown_methods:
        issues.append(f"methods contains unrecognized entries, which is rejected as a safety precaution: {sorted(unknown_methods)}")

    for method_name in REQUIRED_METHOD_NAMES & set(methods.keys()):
        metrics = methods[method_name]
        missing_keys = REQUIRED_METHOD_METRIC_KEYS - set(metrics.keys())
        if missing_keys:
            issues.append(f"methods.{method_name} is missing keys: {sorted(missing_keys)}")
            continue

        unexpected_keys = set(metrics.keys()) - REQUIRED_METHOD_METRIC_KEYS
        if unexpected_keys:
            issues.append(f"methods.{method_name} contains unexpected key(s), which is rejected as a safety precaution: {sorted(unexpected_keys)}")

        for field in RATE_FIELDS:
            value = metrics.get(field)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                issues.append(f"methods.{method_name}.{field} must be a finite number in [0, 1], got {value!r}")

        cm = metrics.get("confusion_matrix", {})
        for key in ("tn", "fp", "fn", "tp"):
            value = cm.get(key)
            if not isinstance(value, int) or value < 0:
                issues.append(f"methods.{method_name}.confusion_matrix.{key} must be a non-negative integer, got {value!r}")

        for count_field in ("predicted_review_case_count", "seasonal_sale_false_positive_count", "early_hidden_risk_false_negative_count"):
            value = metrics.get(count_field)
            if not isinstance(value, int) or value < 0:
                issues.append(f"methods.{method_name}.{count_field} must be a non-negative integer, got {value!r}")

    scenario_difficulty = report.get("scenario_difficulty")
    if not isinstance(scenario_difficulty, list) or not scenario_difficulty:
        issues.append("scenario_difficulty must be a non-empty list")
    else:
        known_labels = set(SCENARIO_LABEL_MAP.values())
        for entry in scenario_difficulty:
            if not isinstance(entry, dict) or {"state", "row_count", "positive_rate", "methods"} - set(entry.keys()):
                issues.append(f"scenario_difficulty entry is missing required keys: {entry!r}")
                continue
            if entry["state"] not in known_labels:
                issues.append(f"scenario_difficulty entry has an unrecognized state label: {entry['state']!r}")
            if not isinstance(entry["row_count"], int) or entry["row_count"] < 0:
                issues.append(f"scenario_difficulty.{entry['state']}.row_count must be a non-negative integer")
            for method_name, breakdown in entry.get("methods", {}).items():
                for rate_field in ("predicted_positive_rate", "recall_within_state"):
                    value = breakdown.get(rate_field)
                    if value is not None and (not isinstance(value, (int, float)) or not (0 <= value <= 1)):
                        issues.append(f"scenario_difficulty.{entry['state']}.methods.{method_name}.{rate_field} must be a finite number in [0, 1] or null, got {value!r}")

    # Check the field-name-documentation-only strings everywhere EXCEPT
    # model.excluded_fields, their one legitimate, required location.
    report_without_excluded_fields_doc = json.loads(json.dumps(report))
    report_without_excluded_fields_doc.get("model", {}).pop("excluded_fields", None)
    serialized_excluding_doc_list = json.dumps(report_without_excluded_fields_doc)
    for field_name in FIELD_NAME_DOCUMENTATION_ONLY:
        if field_name in serialized_excluding_doc_list:
            issues.append(f"Report references {field_name!r} outside of model.excluded_fields -- possible value leak.")

    serialized = json.dumps(report)
    for prohibited in PROHIBITED_LITERAL_STRINGS:
        if prohibited in serialized:
            issues.append(f"Report contains prohibited string: {prohibited!r}")
    if MERCHANT_ID_PATTERN.search(serialized):
        issues.append("Report contains what looks like a merchant ID; the report must be aggregate-only.")

    return issues


def is_valid_report(report: dict) -> bool:
    return len(validate_report(report)) == 0
