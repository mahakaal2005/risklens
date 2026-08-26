"""Review Queue page: summary cards, filters, and a compact case table
sourced entirely from GET /cases. No business logic (state-machine rules,
scoring, etc.) lives here -- only display and selection.
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import ClearRiskAPIClient, DashboardAPIError
from dashboard.components.common import format_timestamp, render_error

STATUS_OPTIONS = ["All", "OPEN", "EVIDENCE_REQUESTED", "EVIDENCE_SUBMITTED", "UNDER_REVIEW", "RESOLVED", "ESCALATED"]
RECOMMENDATION_OPTIONS = ["All", "APPROVE", "ALLOW_WITH_MONITORING", "REQUEST_EVIDENCE", "MANUAL_REVIEW_REQUIRED", "ESCALATE_TO_COMPLIANCE"]
INTENSITY_OPTIONS = ["All", "Low", "Medium", "High"]

MAX_FETCH_LIMIT = 100


def render_review_queue(client: ClearRiskAPIClient) -> None:
    st.title("Review Queue")
    st.caption("Find and prioritize open review cases.")

    try:
        all_cases_response = client.list_cases(limit=MAX_FETCH_LIMIT)
    except DashboardAPIError as exc:
        render_error(exc)
        return

    all_items = all_cases_response.get("items", [])
    status_counts = {
        "OPEN": 0, "EVIDENCE_REQUESTED": 0, "EVIDENCE_SUBMITTED": 0,
        "UNDER_REVIEW": 0, "RESOLVED": 0, "ESCALATED": 0,
    }
    for item in all_items:
        status = item.get("case_status")
        if status in status_counts:
            status_counts[status] += 1

    cols = st.columns(4)
    cols[0].metric("Open", status_counts["OPEN"])
    cols[1].metric("Evidence requested", status_counts["EVIDENCE_REQUESTED"])
    cols[2].metric("Escalated", status_counts["ESCALATED"])
    cols[3].metric("Resolved", status_counts["RESOLVED"])
    st.caption(f"{len(all_items)} case(s) returned (up to {MAX_FETCH_LIMIT} per page).")

    with st.expander("Filters"):
        filter_cols = st.columns(3)
        status_filter = filter_cols[0].selectbox("Case status", STATUS_OPTIONS)
        recommendation_filter = filter_cols[1].selectbox("Recommended workflow action", RECOMMENDATION_OPTIONS)
        intensity_filter = filter_cols[2].selectbox("Risk signal intensity", INTENSITY_OPTIONS)

    try:
        filtered_response = client.list_cases(
            status=None if status_filter == "All" else status_filter,
            recommendation=None if recommendation_filter == "All" else recommendation_filter,
            intensity=None if intensity_filter == "All" else intensity_filter,
            limit=MAX_FETCH_LIMIT,
        )
    except DashboardAPIError as exc:
        render_error(exc)
        return

    items = filtered_response.get("items", [])

    if not items:
        st.info("No cases match the current filters.")
        return

    table_rows = [
        {
            "Case ID": item["case_id"],
            "Merchant ID": item["merchant_id"],
            "Week start": item["week_start"],
            "Risk signal intensity": item["risk_signal_intensity"],
            "Recommended workflow action": item["recommendation"],
            "Case status": item["case_status"],
            "Final outcome": item.get("final_outcome") or "—",
            "Created": format_timestamp(item["created_at"]),
        }
        for item in items
    ]
    selection = st.dataframe(
        table_rows,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="review_queue_table",
    )

    selected_rows = selection.selection.rows if selection and selection.selection else []
    if not selected_rows:
        st.caption("Select a row to open that case.")
        return

    picked = items[selected_rows[0]]
    st.markdown(
        f"**{picked['case_id']}** — {picked['merchant_id']}, week of {picked['week_start']} "
        f"· {picked['case_status']} · {picked['recommendation']}"
    )
    if st.button("Open case detail", type="primary"):
        st.session_state["selected_case_id"] = picked["case_id"]
        st.session_state["nav_page"] = "Case Detail"
        st.rerun()
