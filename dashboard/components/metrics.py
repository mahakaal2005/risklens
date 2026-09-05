"""Overview page: hero section, safety callout, model-comparison cards read
from GET /metrics, and the honest synthetic-data limitation. No metric
values are ever fabricated here -- if the API reports metrics as
unavailable, this page shows that plainly instead of fake numbers.
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import ClearRiskAPIClient, DashboardAPIError
from dashboard.components.common import render_error
from dashboard.components.economics import render_calibration_panel, render_cost_panel

PRODUCT_EXPLANATION = (
    "RiskLens identifies synthetic merchant patterns associated with rising "
    "refund and chargeback loss. It explains the risk, guides evidence collection, "
    "keeps a human reviewer in control, and records an auditable decision trail."
)

THRESHOLD_EXPLANATION = (
    "The current threshold favors recall for early-warning review routing. "
    "It is not an automatic enforcement threshold."
)

HONEST_LIMITATION = (
    "Synthetic-data metrics demonstrate the prototype workflow only. They do not prove "
    "real-world chargeback-risk performance."
)

METHOD_GUIDANCE = [
    ("Logistic Regression", "Lower false-positive rate, fully interpretable — the model actually used for live case scoring."),
    ("Random Forest", "Comparison baseline only (not used for live scoring) — shown to check whether added complexity earns better held-out performance."),
    ("Gradient Boosting", "Comparison baseline only (not used for live scoring) — same purpose as Random Forest, a different model family."),
    ("Trajectory Transformer", "Comparison baseline only (not used for live scoring) — reads the merchant's last 8 weeks instead of one week, to test whether the trend itself carries signal."),
    ("Combined policy", "Higher recall — best for a conservative risk-operations queue; the reviewer handles more cases. Built on Logistic Regression, the live-scoring model."),
    ("Rules-only", "Fully transparent fallback, but lower measured performance on synthetic test data."),
]


METHOD_ROWS = [
    ("Rules-only", "rules_only_metrics"),
    ("Logistic Regression", "logistic_regression_metrics"),
    ("Random Forest", "random_forest_metrics"),
    ("Gradient Boosting", "gradient_boosting_metrics"),
    ("Trajectory Transformer", "trajectory_transformer_metrics"),
    ("Combined policy", "combined_policy_metrics"),
]


def _fmt(metrics: dict, key: str) -> str:
    """Format a metric, or an em dash when the API did not provide it.
    Never substitutes 0.000 for a missing value."""
    value = metrics.get(key)
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


def _comparison_rows(metrics: dict) -> list[dict]:
    rows = []
    for label, response_key in METHOD_ROWS:
        method_metrics = metrics.get(response_key) or {}
        rows.append(
            {
                "Method": label,
                "Precision": _fmt(method_metrics, "precision"),
                "Recall": _fmt(method_metrics, "recall"),
                "PR-AUC": _fmt(method_metrics, "pr_auc"),
                "False-positive rate": _fmt(method_metrics, "false_positive_rate"),
            }
        )
    return rows


def _render_at_a_glance(client: ClearRiskAPIClient) -> None:
    """Quick case-load summary so the Overview page opens with something
    concrete, not just prose -- mirrors the Review Queue's summary cards
    using the same list_cases() call, no new endpoint needed."""
    try:
        cases_response = client.list_cases(limit=100)
    except DashboardAPIError:
        return  # Overview's own model-metrics section still renders; this is a bonus strip.

    items = cases_response.get("items", [])
    if not items:
        return

    open_count = sum(1 for item in items if item.get("case_status") == "OPEN")
    escalated_count = sum(1 for item in items if item.get("case_status") == "ESCALATED")
    breached_count = sum(1 for item in items if item.get("sla_breached"))

    cols = st.columns(4)
    cols[0].metric("Total cases", len(items))
    cols[1].metric("Open", open_count)
    cols[2].metric("Escalated", escalated_count)
    cols[3].metric("SLA breached", breached_count)


def render_overview(client: ClearRiskAPIClient) -> None:
    with st.container(border=True):
        st.title("RiskLens")
        st.subheader("Explainable early warning for merchant refund and chargeback spikes.")
        st.write(PRODUCT_EXPLANATION)

    st.markdown("#### At a glance")
    _render_at_a_glance(client)

    st.markdown("#### Model comparison — held-out synthetic test data")

    try:
        metrics = client.get_metrics()
    except DashboardAPIError as exc:
        render_error(exc)
        return

    if metrics.get("status") != "available":
        st.info(
            f"Evaluation metrics are not available yet. Run this local command to generate them:\n\n"
            f"`{metrics.get('generation_command', 'python3 -m ml.evaluate_model')}`",
            icon="ℹ️",
        )
        st.caption(metrics.get("message", ""))
        return

    with st.container(border=True):
        st.dataframe(_comparison_rows(metrics), width="stretch", hide_index=True)

        threshold = metrics.get("selected_threshold")
        st.caption(
            f"Operating threshold: **{threshold if threshold is not None else '—'}** · {THRESHOLD_EXPLANATION}"
        )

        with st.expander("How to read this comparison"):
            for name, guidance in METHOD_GUIDANCE:
                st.markdown(f"- **{name}**: {guidance}")
            st.caption(
                "False-positive rate is the share of non-elevated merchant-weeks that were still "
                "routed for review — the direct cost of over-flagging good merchants."
            )

    scenario_difficulty = metrics.get("scenario_difficulty")
    if scenario_difficulty:
        with st.expander("Difficulty by scenario"):
            st.caption(
                "How well each method recovers the elevated-loss label within each demonstration "
                "scenario — some scenarios are supposed to be harder to catch by design (see MODEL_CARD.md)."
            )
            scenario_rows = []
            for entry in scenario_difficulty:
                row = {"Scenario": entry["state"], "Rows": entry["row_count"], "Positive rate": entry["positive_rate"]}
                for method_name, breakdown in entry.get("methods", {}).items():
                    row[f"{method_name} recall"] = breakdown.get("recall_within_state")
                scenario_rows.append(row)
            st.dataframe(scenario_rows, width="stretch", hide_index=True)

    render_cost_panel(metrics)
    render_calibration_panel(metrics)

    st.warning(HONEST_LIMITATION, icon="📊")

    with st.expander("Scope and safety boundaries"):
        st.markdown(
            "- Synthetic data only\n"
            "- Review recommendations only\n"
            "- No payment or account enforcement\n"
            "- No real gateway integration"
        )
