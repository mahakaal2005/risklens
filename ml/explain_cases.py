"""Milestone 4: deterministic, template/rules-based explanation generation.

No LLM call and no agent is used anywhere in this module -- every sentence
is produced from a fixed template filled in with actual model/rule values.
Top model factors are computed directly from the fitted Logistic Regression
coefficients and each row's transformed feature values, so the explanation
only ever references information the model actually used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_LABELS = {
    "refund_rate_change": "refund-rate change",
    "chargeback_rate_change": "chargeback-rate change",
    "transaction_volume_change": "transaction-volume change",
    "refund_to_chargeback_ratio": "refund-to-chargeback ratio",
    "delivery_evidence_gap": "delivery-evidence gap",
    "support_resolution_hours_normalized": "support resolution time",
    "support_ticket_rate": "support ticket rate",
    "merchant_age_days": "merchant age",
    "transaction_count_30d": "transaction count",
}

MAX_TOP_FACTORS = 5

# Explainability-quality policy (added after Milestone 4 review): "A model
# feature may be shown as a natural-language positive or negative risk
# factor only when its observed direction is plausible, stable across
# validation/test or sensitivity checks, and does not contradict the
# documented risk-policy interpretation. Otherwise it is diagnostic-only and
# excluded from ranked explanations. The model may continue to use it, but
# its direction must be documented as unvalidated." See MODEL_CARD.md and
# docs/MILESTONE_4_EXPLAINABILITY.md.
#
# support_ticket_rate has a negative Logistic Regression coefficient (higher
# ticket rate slightly *lowers* the predicted score), which contradicts
# RISK_POLICY.md's SUPPORT_OPERATIONAL_STRESS rule interpretation (rising
# support load is treated as a risk signal, not a risk-reducing one). It has
# not been checked for stability across a sensitivity analysis, so it stays
# diagnostic-only: the model keeps using it, but it is never rendered as a
# named risk factor in analyst or merchant-safe text.
DIAGNOSTIC_ONLY_FEATURES = {"support_ticket_rate"}


def _numeric_sentence(label: str, transformed_value: float, contribution: float) -> str:
    level = "Higher than usual" if transformed_value >= 0 else "Lower than usual"
    effect = "contributed to the elevated score" if contribution > 0 else "contributed to a lower score"
    return f"{level} {label} {effect}."


def _categorical_sentence(group_label: str, category_value: str, contribution: float) -> str:
    effect = "contributed to a higher score" if contribution > 0 else "contributed to a lower score"
    return f"{group_label} '{category_value}' {effect}."


def compute_top_factors(pipeline, row_df: pd.DataFrame, top_n: int = MAX_TOP_FACTORS) -> list[dict]:
    """Returns the top contributing factors for a single-row dataframe, using
    the pipeline's own fitted coefficients and transformed values -- never a
    separate/approximate importance measure.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(row_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)[0]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]
    contributions = coefficients * transformed

    factors = []
    for name, value, contribution in zip(feature_names, transformed, contributions):
        if abs(contribution) < 1e-9:
            continue
        if name.startswith("numeric__"):
            raw_name = name.removeprefix("numeric__")
            if raw_name in DIAGNOSTIC_ONLY_FEATURES:
                continue
            label = FEATURE_LABELS.get(raw_name, raw_name.replace("_", " "))
            sentence = _numeric_sentence(label, value, contribution)
            factors.append({"feature": raw_name, "contribution": round(float(contribution), 4), "sentence": sentence})
        elif name.startswith("categorical__"):
            raw_name = name.removeprefix("categorical__")
            if raw_name.startswith("merchant_category_"):
                group_label, category_value = "Merchant category", raw_name.removeprefix("merchant_category_")
            elif raw_name.startswith("previous_review_outcome_"):
                group_label, category_value = "Previous review outcome", raw_name.removeprefix("previous_review_outcome_")
            else:
                group_label, category_value = raw_name, ""
            sentence = _categorical_sentence(group_label, category_value, contribution)
            factors.append({"feature": raw_name, "contribution": round(float(contribution), 4), "sentence": sentence})

    factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)
    return factors[:top_n]


def build_uncertainty_statement(model_probability: float | None, degraded_mode: bool) -> str:
    if degraded_mode:
        return (
            "This assessment used the rules engine only; the Logistic Regression model was "
            "unavailable, so this is a degraded-mode explanation with no model probability."
        )
    return (
        "This assessment reflects synthetic-data model and rule signals as of the prediction "
        "date only; it does not guarantee a future outcome and is not a confirmed finding of fraud."
    )


def build_trend_values(record: dict, features: dict) -> dict:
    return {
        "refund_rate_current_vs_prior": {
            "current": record.get("refund_rate_30d"),
            "prior": record.get("refund_rate_previous_30d"),
        },
        "chargeback_rate_current_vs_prior": {
            "current": record.get("chargeback_rate_30d"),
            "prior": record.get("chargeback_rate_previous_30d"),
        },
        "transaction_volume_current_vs_prior": {
            "current": record.get("transaction_volume_30d"),
            "prior": record.get("transaction_volume_previous_30d"),
        },
        "delivery_evidence_coverage": record.get("delivery_evidence_coverage"),
        "support_ticket_rate": record.get("support_ticket_rate"),
        "support_resolution_time_hours": record.get("average_support_resolution_time_hours"),
    }


def build_analyst_summary(triggered_rule_ids: list[str], top_factors: list[dict], recommendation: str, degraded_mode: bool) -> str:
    sentences = []
    if not triggered_rule_ids and not top_factors:
        sentences.append("No rule or model signal is elevated for this merchant-week.")
    else:
        if triggered_rule_ids:
            sentences.append(f"Triggered rule(s): {', '.join(triggered_rule_ids)}.")
        if top_factors and not degraded_mode:
            sentences.append(top_factors[0]["sentence"])
    sentences.append(f"Recommended action: {recommendation}.")
    if not degraded_mode:
        sentences.append("This reflects rule and model signals only, not a confirmed outcome.")
    return " ".join(sentences[:4])
