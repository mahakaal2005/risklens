"""Cost and calibration panels for the Overview page.

Both read straight from GET /metrics. Neither ever fabricates a number: when
the API serves null for a section (a report generated before that analysis
existed), the panel renders an explicit placeholder rather than zeros, an
error, or a blank space that could be mistaken for "no cost".

Two findings these panels exist to make visible, because a table of numbers
buries them:

1. The best model beats "review everybody" by only ~12%. The bar chart is
   deliberately annotated with that number so the visual does not imply a
   bigger win than the arithmetic supports.
2. Random Forest's probability scale is inflated (~2x the true base rate),
   which is the root cause of the Day 1 threshold-selection bug. The
   reliability curve shows it directly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

COST_PLACEHOLDER = (
    "Rupee cost analysis is not present in the current evaluation report. "
    "Run `python3 -m ml.evaluate_model` to generate it."
)
CALIBRATION_PLACEHOLDER = (
    "Calibration analysis is not present in the current evaluation report. "
    "Run `python3 -m ml.evaluate_model` to generate it."
)

ASSUMPTION_WARNING = (
    "**Every rupee figure below is an assumption, not a measured cost.** They come from "
    "`rules/cost_model.yaml` and exist to show the shape of the tradeoff on synthetic data. "
    "They are not Razorpay's costs and not any real provider's costs."
)

RANDOM_FOREST_CROSS_LINK = (
    "Random Forest's inflated probability scale is why it selected a 0.55 threshold in Day 1 "
    "while every other model chose 0.10–0.15 — its curve sits below the diagonal because it "
    "states roughly double the true risk."
)

CALIBRATION_CAVEAT = (
    "Calibrators are fit on the validation split, which also selects the operating threshold — "
    "validation does double duty. The held-out test set is scored only, never fit on."
)

# Methods drawn on the reliability chart, in draw order. Logistic Regression
# carries the raw-vs-isotonic before/after story; Random Forest raw is included
# because its miscalibration is the cross-link to the threshold bug.
RELIABILITY_SERIES = [
    ("logistic_regression", "raw", "Logistic Regression (raw)"),
    ("logistic_regression", "isotonic", "Logistic Regression (isotonic)"),
    ("random_forest", "raw", "Random Forest (raw)"),
]

CALIBRATION_TABLE_METHODS = [
    ("logistic_regression", "Logistic Regression"),
    ("random_forest", "Random Forest"),
    ("gradient_boosting", "Gradient Boosting"),
    ("trajectory_transformer", "Trajectory Transformer"),
]


def _crore(rupees: float) -> str:
    """Indian-convention formatting: rupee amounts in this range read far more
    naturally in crore than as a 9-digit figure."""
    return f"₹{rupees / 10_000_000:.2f} Cr"


def _cost_bar_frame(headline: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Policy": [
                "Rules-only",
                "Review everybody",
                f"Best model ({headline['best_model_name'].replace('_', ' ').title()})",
            ],
            "Expected cost (₹)": [
                headline["rules_only_cost_inr"],
                headline["review_everybody_cost_inr"],
                headline["best_model_cost_inr"],
            ],
        }
    ).set_index("Policy")


def render_cost_panel(metrics: dict) -> None:
    cost_analysis = metrics.get("cost_analysis")
    headline = (cost_analysis or {}).get("headline_comparison")
    if not headline:
        st.info(COST_PLACEHOLDER, icon="ℹ️")
        return

    st.markdown("#### What it costs to be wrong — held-out synthetic test data")
    st.warning(ASSUMPTION_WARNING, icon="⚠️")

    chart_column, text_column = st.columns([3, 2])

    with chart_column:
        st.bar_chart(_cost_bar_frame(headline), horizontal=True, height=220)

    with text_column:
        margin = headline["best_model_margin_over_review_everybody_pct"]
        st.metric(
            "Best model vs. reviewing everybody",
            f"{margin:.1f}% cheaper",
            help="Lower expected rupee cost than simply reviewing every merchant-week.",
        )
        st.markdown(
            f"- Rules-only: **{_crore(headline['rules_only_cost_inr'])}**\n"
            f"- Review everybody: **{_crore(headline['review_everybody_cost_inr'])}**\n"
            f"- Best model: **{_crore(headline['best_model_cost_inr'])}**"
        )

    # The two findings the bars alone would misrepresent.
    if not headline["rules_only_beats_review_everybody"]:
        st.error(
            "**The rules-only baseline costs more than simply reviewing every merchant-week.** "
            "Under these assumed costs it adds negative value against that baseline — every ML "
            "method beats it, but the transparent fallback on its own does not.",
            icon="🚩",
        )
    st.caption(
        f"The model's margin over reviewing everybody is only **{headline['best_model_margin_over_review_everybody_pct']:.1f}%**. "
        "At the assumed 72:1 miss-to-review cost ratio most of the value comes from reviewing a lot, "
        "not from the model's discrimination. Compare the bars against each other, not against zero."
    )

    with st.expander("Cost assumptions and sensitivity"):
        assumptions = cost_analysis.get("cost_assumptions_inr", {})
        st.markdown(
            f"- Review cost: ₹{assumptions.get('review_cost', '—')}\n"
            f"- Missed loss: ₹{assumptions.get('missed_loss', '—')}\n"
            f"- False-positive friction: ₹{assumptions.get('false_positive_friction', '—')}\n"
            f"- Recovery rate: {cost_analysis.get('recovery_rate', '—')}"
        )
        sensitivity = cost_analysis.get("threshold_sensitivity_logistic_regression")
        if sensitivity:
            st.dataframe(pd.DataFrame(sensitivity), width="stretch", hide_index=True)
            st.caption(
                "How the cost-optimal threshold moves with the assumed cost ratio. Above roughly "
                "50:1 the recommendation saturates, so the specific ₹18,000 assumption does less "
                "work than it appears."
            )
        st.caption(cost_analysis.get("assumption_notice", ""))


def _reliability_frame(calibration: dict) -> pd.DataFrame | None:
    """Wide frame of observed positive rate per predicted-probability bin, one
    column per drawn series, plus the y=x reference. Empty bins are dropped
    rather than plotted as zero -- a bin with no predictions has no observed
    rate, and drawing 0.0 there would invent a data point."""
    methods = calibration.get("methods") or {}
    series = {}

    for method_key, variant, label in RELIABILITY_SERIES:
        curve = (
            methods.get(method_key, {})
            .get("metrics", {})
            .get(variant, {})
            .get("reliability_curve")
        )
        if not curve:
            continue
        points = {
            entry["mean_predicted_probability"]: entry["observed_positive_rate"]
            for entry in curve
            if entry.get("count") and entry.get("mean_predicted_probability") is not None
        }
        if points:
            series[label] = points

    if not series:
        return None

    frame = pd.DataFrame(series).sort_index()
    frame.index.name = "Predicted probability"
    # Reference line: a perfectly calibrated model sits exactly on y = x.
    frame["Perfect calibration (y=x)"] = frame.index
    return frame


def _calibration_table(calibration: dict) -> list[dict]:
    methods = calibration.get("methods") or {}
    rows = []
    for method_key, label in CALIBRATION_TABLE_METHODS:
        metrics = (methods.get(method_key) or {}).get("metrics") or {}
        for variant in ("raw", "isotonic"):
            values = metrics.get(variant)
            if not values:
                continue
            rows.append({
                "Method": label,
                "Variant": variant,
                "Brier": f"{values['brier_score']:.4f}",
                "ECE": f"{values['expected_calibration_error']:.4f}",
                "MCE": f"{values['maximum_calibration_error']:.4f}",
                "Mean predicted": f"{values['mean_predicted_probability']:.4f}",
                "Observed": f"{values['observed_positive_rate']:.4f}",
            })
    return rows


def render_calibration_panel(metrics: dict) -> None:
    calibration = metrics.get("calibration")
    if not calibration:
        st.info(CALIBRATION_PLACEHOLDER, icon="ℹ️")
        return

    st.markdown("#### Is the stated probability honest?")
    st.caption(
        "Ranking quality (PR-AUC) and calibration are different properties. This asks: among "
        "merchant-weeks scored 0.30, do roughly 30% actually deteriorate?"
    )

    frame = _reliability_frame(calibration)
    if frame is not None:
        st.line_chart(frame, height=320)
        st.caption(
            "Reliability curve — predicted probability (x) against observed positive rate (y). "
            "A perfectly calibrated model follows the dashed y=x reference. Points **above** the "
            "line understate risk; points **below** overstate it."
        )
        st.info(RANDOM_FOREST_CROSS_LINK, icon="🔗")
    else:
        st.caption("Reliability-curve bin data is not present in this report; summary metrics only.")

    table = _calibration_table(calibration)
    if table:
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption(
            "Brier is a proper scoring rule and is the primary number. ECE is the count-weighted "
            "average calibration gap; MCE is the worst single bin — reported because ECE's "
            "weighting hides errors in the sparse high-probability bins that produce escalations."
        )

    st.caption(calibration.get("methodology_note", CALIBRATION_CAVEAT))
