"""Shared utilities for Milestone 3: feature-column definitions, the
preprocessing pipeline, the rules-only baseline definition, the combined
model+rules comparison policy, and evaluation metrics / the near-perfect
investigation gate.

Nothing here is a production enforcement policy -- combined_policy() is a
decision-support comparison only, and its only outputs are the five allowed
recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.features import FEATURE_CATALOGUE

NUMERIC_ML_FEATURES = [f["name"] for f in FEATURE_CATALOGUE if f["used_by"] in ("ml", "both") and f["name"] not in ("merchant_category", "previous_review_outcome")]
CATEGORICAL_ML_FEATURES = ["merchant_category", "previous_review_outcome"]
ML_FEATURE_COLUMNS = NUMERIC_ML_FEATURES + CATEGORICAL_ML_FEATURES

ALLOWED_RECOMMENDATIONS = {
    "APPROVE",
    "ALLOW_WITH_MONITORING",
    "REQUEST_EVIDENCE",
    "MANUAL_REVIEW_REQUIRED",
    "ESCALATE_TO_COMPLIANCE",
}

RULES_ONLY_POSITIVE_TIERS = {"medium", "high"}
RULES_ONLY_POSITIVE_RECOMMENDATIONS = {"REQUEST_EVIDENCE", "MANUAL_REVIEW_REQUIRED", "ESCALATE_TO_COMPLIANCE"}

CHARGEBACK_OR_COMBINED_RULES = {"CHARGEBACK_RATE_SPIKE", "COMBINED_LOSS_SIGNAL"}
OTHER_MEDIUM_RULES = {"REFUND_RATE_SPIKE", "EVIDENCE_COVERAGE_GAP", "SUPPORT_OPERATIONAL_STRESS"}

NEAR_PERFECT_PR_AUC = 0.98
NEAR_PERFECT_PRECISION = 0.98
NEAR_PERFECT_RECALL = 0.98

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
FBETA_BETA = 2.0
MIN_PRECISION = 0.30
MAX_FALSE_POSITIVE_RATE = 0.20


def build_preprocessing_pipeline() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_ML_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_ML_FEATURES),
    ])


def rules_only_positive_prediction(rules_result: dict) -> bool:
    """Defines the rules-only baseline's binary "review needed" prediction.

    Positive (review needed) when the rules-only tier is medium/high, OR the
    recommendation is REQUEST_EVIDENCE, MANUAL_REVIEW_REQUIRED, or
    ESCALATE_TO_COMPLIANCE. This definition is fixed before looking at any
    held-out result and is not tuned against test-set outcomes.
    """
    return (
        rules_result["risk_tier"] in RULES_ONLY_POSITIVE_TIERS
        or rules_result["recommendation"] in RULES_ONLY_POSITIVE_RECOMMENDATIONS
    )


@dataclass(frozen=True)
class CombinedDecision:
    recommendation: str
    risk_signal_intensity: str
    policy_explanation: str


def combined_policy(ml_probability: float, selected_threshold: float, triggered_rules: set[str]) -> CombinedDecision:
    """Comparison-only combined model+rules policy. Never returns an
    enforcement outcome -- only the five allowed recommendations.

    ML risk band: 'high' at/above the selected operating threshold, 'medium'
    from half that threshold up to it, else 'low'. This reuses the single
    already-selected, validation-derived threshold rather than inventing a
    second unvalidated cut point.
    """
    low_threshold = selected_threshold / 2
    if ml_probability >= selected_threshold:
        ml_band = "high"
    elif ml_probability >= low_threshold:
        ml_band = "medium"
    else:
        ml_band = "low"

    chargeback_or_combined = bool(CHARGEBACK_OR_COMBINED_RULES & triggered_rules)
    other_medium = bool(OTHER_MEDIUM_RULES & triggered_rules)

    if chargeback_or_combined and ml_band == "high":
        recommendation = "MANUAL_REVIEW_REQUIRED"
        explanation = (
            "High ML risk together with a chargeback-spike or combined-loss rule trigger "
            "indicates a serious, model-and-rule-confirmed pattern."
        )
    elif chargeback_or_combined:
        recommendation = "MANUAL_REVIEW_REQUIRED"
        explanation = (
            "A chargeback-spike or combined-loss rule triggered regardless of the model's "
            "probability band; this rule evidence is treated as sufficient on its own, "
            "consistent with its Milestone 2 standalone severity."
        )
    elif ml_band == "high" and other_medium:
        recommendation = "REQUEST_EVIDENCE"
        explanation = (
            "High model risk combined with operational-stress/evidence-gap signals, but no "
            "confirmed chargeback-spike or combined-loss rule, so evidence is requested "
            "rather than escalated to manual review -- treated as a potentially remediable "
            "operational problem rather than confirmed abuse."
        )
    elif ml_band == "high":
        recommendation = "MANUAL_REVIEW_REQUIRED"
        explanation = (
            "The model alone flags high risk with no supporting rule trigger. This is a "
            "model/rules disagreement, and is routed to manual review rather than approved "
            "or dismissed."
        )
    elif ml_band == "medium" and other_medium:
        recommendation = "REQUEST_EVIDENCE"
        explanation = (
            "Medium model risk alongside a refund/evidence/support signal (e.g. a refund-only "
            "seasonal-sale-like pattern) is treated as reviewable, not high-risk."
        )
    elif ml_band == "medium":
        recommendation = "ALLOW_WITH_MONITORING"
        explanation = "Medium model risk with no supporting rule signal; monitored rather than actioned."
    elif triggered_rules:
        recommendation = "ALLOW_WITH_MONITORING"
        explanation = "Low model risk with a minor rule signal; monitored rather than actioned."
    else:
        recommendation = "APPROVE"
        explanation = "Low model risk and no rule triggered."

    if ml_band == "high" or chargeback_or_combined:
        risk_signal_intensity = "High"
    elif ml_band == "medium" or other_medium:
        risk_signal_intensity = "Medium"
    else:
        risk_signal_intensity = "Low"

    assert recommendation in ALLOWED_RECOMMENDATIONS
    return CombinedDecision(recommendation, risk_signal_intensity, explanation)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """y_score is a ranking signal in [0, 1] used for PR-AUC/ROC-AUC. For the
    rules-only baseline this is risk_score/100 (a monotonic severity surrogate,
    not a calibrated probability) -- documented explicitly wherever reported.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "prevalence": round(float(np.mean(y_true)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(fbeta_score(y_true, y_pred, beta=1.0, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
        "false_negative_rate": round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_predicted_positive": int(np.sum(y_pred)),
    }
    if y_score is not None:
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_score)), 4)
        metrics["roc_auc_secondary"] = round(float(roc_auc_score(y_true, y_score)), 4)
    else:
        metrics["pr_auc"] = None
        metrics["roc_auc_secondary"] = None
    return metrics


def select_threshold(y_val: np.ndarray, val_probs: np.ndarray) -> dict:
    """Selects the validation threshold maximizing F-beta(2), subject to
    precision >= MIN_PRECISION and false-positive rate <= MAX_FALSE_POSITIVE_RATE.
    Falls back to the unconstrained F-beta(2) maximizer, clearly flagged, if
    no threshold satisfies both constraints. Held-out test data is never
    used here. Shared by every training script (Logistic Regression,
    Random Forest, Gradient Boosting) so threshold-selection logic is
    defined once, not copied per model.
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


def compute_scenario_difficulty(df: pd.DataFrame, latent_state_column: str, label_column: str, method_predictions: dict[str, np.ndarray]) -> list[dict]:
    """Per-latent-state difficulty breakdown for the held-out test set.

    Generalizes the ad hoc seasonal_false_positives/early_hidden_false_negatives
    counts into a systematic per-scenario view: for every latent state
    actually present, report how prevalent the positive label is within
    that state, and how well each method (rules-only, each ML model,
    combined) recovers those positives -- some states (e.g.
    early_hidden_risk) are supposed to be hard to catch by design, and this
    makes that difficulty visible rather than averaging it away in one
    aggregate number.
    """
    results = []
    for state in sorted(df[latent_state_column].unique()):
        state_mask = (df[latent_state_column] == state).to_numpy()
        row_count = int(state_mask.sum())
        y_state = df.loc[state_mask, label_column].to_numpy()
        positive_count = int(y_state.sum())
        positive_rate = round(float(y_state.mean()), 4) if row_count else None

        method_breakdown = {}
        for method_name, y_pred in method_predictions.items():
            pred_state = np.asarray(y_pred)[state_mask]
            predicted_positive_rate = round(float(pred_state.mean()), 4) if row_count else None
            if positive_count > 0:
                recall_within_state = round(float(pred_state[y_state == 1].mean()), 4)
            else:
                recall_within_state = None
            method_breakdown[method_name] = {
                "predicted_positive_rate": predicted_positive_rate,
                "recall_within_state": recall_within_state,
            }

        results.append({
            "state": state,
            "row_count": row_count,
            "positive_rate": positive_rate,
            "methods": method_breakdown,
        })
    return results


def check_near_perfect(metrics: dict) -> tuple[bool, str]:
    cm = metrics["confusion_matrix"]
    zero_fp_or_fn = cm["fp"] == 0 or cm["fn"] == 0
    pr_auc_flag = metrics["pr_auc"] is not None and metrics["pr_auc"] >= NEAR_PERFECT_PR_AUC
    precision_recall_flag = metrics["precision"] >= NEAR_PERFECT_PRECISION and metrics["recall"] >= NEAR_PERFECT_RECALL

    if pr_auc_flag or precision_recall_flag or zero_fp_or_fn:
        reasons = []
        if pr_auc_flag:
            reasons.append(f"PR-AUC {metrics['pr_auc']} >= {NEAR_PERFECT_PR_AUC}")
        if precision_recall_flag:
            reasons.append(f"precision {metrics['precision']} and recall {metrics['recall']} both >= 0.98")
        if zero_fp_or_fn:
            reasons.append(f"zero false positives or false negatives (fp={cm['fp']}, fn={cm['fn']})")
        return True, "; ".join(reasons)
    return False, "No near-perfect condition triggered."
