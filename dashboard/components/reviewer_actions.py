"""Reviewer Actions component: only offers actions valid for the case's
current status, matching the state machine already enforced by
app/services/case_service.py. This component never decides validity itself
-- it only limits which buttons are shown, and the backend is always the
final authority (an invalid action still safely surfaces a 409).
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import ClearRiskAPIClient, DashboardAPIError
from dashboard.components.common import format_timestamp, render_error

# case_status -> list of (action_value, safe_label)
ACTIONS_BY_STATUS = {
    "OPEN": [
        ("REQUEST_EVIDENCE", "Request evidence"),
        ("CLEAR_CASE", "Clear case"),
        ("MARK_FALSE_POSITIVE", "Mark false positive"),
        ("MARK_OPERATIONAL_ISSUE", "Mark operational issue"),
        ("MARK_INCONCLUSIVE", "Mark inconclusive"),
        ("ESCALATE_CASE", "Escalate case"),
    ],
    "EVIDENCE_REQUESTED": [
        ("ESCALATE_CASE", "Escalate case"),
    ],
    "EVIDENCE_SUBMITTED": [
        ("START_REVIEW", "Start review"),
        ("ESCALATE_CASE", "Escalate case"),
    ],
    "UNDER_REVIEW": [
        ("CLEAR_CASE", "Clear case"),
        ("MARK_FALSE_POSITIVE", "Mark false positive"),
        ("MARK_OPERATIONAL_ISSUE", "Mark operational issue"),
        ("MARK_INCONCLUSIVE", "Mark inconclusive"),
        ("ESCALATE_CASE", "Escalate case"),
    ],
}


def render_reviewer_actions(client: ClearRiskAPIClient, case: dict) -> None:
    st.markdown("### Reviewer actions")
    status = case.get("case_status")

    if status == "RESOLVED":
        st.success("This case is resolved. It is immutable — no further reviewer action is available.")
        st.markdown(f"**Final outcome:** {case.get('final_outcome') or '—'}")
        st.markdown(f"**Reviewer note:** {case.get('reviewer_note') or '—'}")
        st.markdown(f"**Resolved at:** {format_timestamp(case.get('resolved_at'))}")
        return

    if status == "ESCALATED":
        st.warning("This case is escalated for compliance review. No automated action is available here.")
        return

    if status == "EVIDENCE_REQUESTED":
        st.info("Waiting for merchant evidence.")

    available_actions = ACTIONS_BY_STATUS.get(status, [])
    if not available_actions:
        st.caption("No reviewer actions are available for this case status.")
        return

    action_labels = [label for _, label in available_actions]
    action_values = {label: value for value, label in available_actions}

    selected_label = st.selectbox("Select an action", action_labels, key=f"action_select_{case['case_id']}")
    note = st.text_area("Reviewer note (required)", key=f"reviewer_note_{case['case_id']}")

    if st.button("Submit action", type="primary", key=f"submit_action_{case['case_id']}"):
        if not note or not note.strip():
            st.error("A reviewer note is required before submitting an action.")
            return
        action_value = action_values[selected_label]
        try:
            client.submit_review_action(case["case_id"], action_value, note)
            st.success(f"Action '{selected_label}' recorded.")
            st.rerun()
        except DashboardAPIError as exc:
            render_error(exc)
