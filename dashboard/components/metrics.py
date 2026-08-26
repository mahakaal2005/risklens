"""Overview page: hero section, safety callout, model-comparison cards read
from GET /metrics, and the honest synthetic-data limitation. No metric
values are ever fabricated here -- if the API reports metrics as
unavailable, this page shows that plainly instead of fake numbers.
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import ClearRiskAPIClient, DashboardAPIError
from dashboard.components.common import render_error

PRODUCT_EXPLANATION = (
    "ClearRisk Recover identifies synthetic merchant patterns associated with rising "
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
    ("Logistic Regression", "Lower false-positive rate — best for a low-friction early-warning dashboard."),
    ("Combined policy", "Higher recall — best for a conservative risk-operations queue; the reviewer handles more cases."),
    ("Rules-only", "Fully transparent fallback, but lower measured performance on synthetic test data."),
]


METHOD_ROWS = [
    ("Rules-only", "rules_only_metrics"),
    ("Logistic Regression", "logistic_regression_metrics"),
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


def render_overview(client: ClearRiskAPIClient) -> None:
    st.title("ClearRisk Recover")
    st.subheader("Explainable early warning for merchant refund and chargeback spikes.")
    st.write(PRODUCT_EXPLANATION)

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

    st.warning(HONEST_LIMITATION, icon="📊")

    with st.expander("Scope and safety boundaries"):
        st.markdown(
            "- Synthetic data only\n"
            "- Review recommendations only\n"
            "- No payment or account enforcement\n"
            "- No real gateway integration"
        )
