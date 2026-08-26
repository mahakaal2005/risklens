"""Audit Timeline page: the ordered, append-only audit trail for a case,
read from GET /cases/{case_id}/audit-events. Raw payload JSON is hidden by
default behind an explicit "View safe event details" expander.
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import ClearRiskAPIClient, DashboardAPIError
from dashboard.components.common import format_timestamp, render_error

ACTOR_LABELS = {
    "system": "🖥️ System event",
    "analyst_demo": "🧑‍💼 Reviewer event",
    "merchant_demo": "🏬 Merchant event",
}

APPEND_ONLY_NOTICE = (
    "This is an append-only application audit log for a local prototype. "
    "It is not cryptographically immutable storage."
)


def _actor_label(actor_type: str) -> str:
    return ACTOR_LABELS.get(actor_type, f"Event ({actor_type})")


def render_audit_timeline(client: ClearRiskAPIClient) -> None:
    st.title("Audit Timeline")
    st.caption("Make the product's fairness and traceability visible.")

    try:
        cases_response = client.list_cases(limit=100)
        available_case_ids = [item["case_id"] for item in cases_response.get("items", [])]
    except DashboardAPIError as exc:
        render_error(exc)
        return

    if not available_case_ids:
        st.info("No cases exist yet. Seed demo cases first (see docs/UI_DEMO_GUIDE.md).")
        return

    default_index = 0
    selected_case_id = st.session_state.get("selected_case_id")
    if selected_case_id in available_case_ids:
        default_index = available_case_ids.index(selected_case_id)
    picked = st.selectbox("Case ID", available_case_ids, index=default_index)

    try:
        timeline_response = client.get_audit_events(picked)
    except DashboardAPIError as exc:
        render_error(exc)
        return

    events = sorted(timeline_response.get("events", []), key=lambda e: e["event_sequence_number"])

    if not events:
        st.info("No audit events recorded for this case yet.")
        return

    timeline_rows = [
        {
            "#": event["event_sequence_number"],
            "Timestamp": format_timestamp(event["event_timestamp"]),
            "Actor": _actor_label(event["actor_type"]),
            "Event": event["event_type"],
        }
        for event in events
    ]

    selection = st.dataframe(
        timeline_rows,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"audit_timeline_table_{picked}",
    )

    selected_rows = selection.selection.rows if selection and selection.selection else []
    if selected_rows:
        event = events[selected_rows[0]]
        st.markdown(f"**Event #{event['event_sequence_number']} — {event['event_type']}**")
        payload = event.get("event_payload") or {}
        if payload:
            for key, value in payload.items():
                st.markdown(f"- **{key}**: {value}")
        else:
            st.caption("No additional payload for this event.")
    else:
        st.caption("Select a row to see that event's safe payload detail.")

    st.info(APPEND_ONLY_NOTICE, icon="🔒")
