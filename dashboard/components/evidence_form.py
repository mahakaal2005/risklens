"""Merchant Response page: simulates a merchant responding to an evidence
request. No real file upload or URL fetching exists here -- evidence
references are validated strings only, submitted through the existing
POST /cases/{case_id}/evidence endpoint.
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import ClearRiskAPIClient, DashboardAPIError
from dashboard.components.common import render_error

EXAMPLE_REFERENCES = [
    "invoice_demo_001.pdf",
    "delivery_proof_demo_001.pdf",
    "refund_records_demo_001.pdf",
    "seasonal_sale_summary_demo_001.txt",
]


def render_merchant_response(client: ClearRiskAPIClient) -> None:
    st.title("Merchant Response")
    st.caption("Simulates a merchant submitting a response to a review case. No real file upload exists.")

    try:
        cases_response = client.list_cases(limit=100)
        available_case_ids = [item["case_id"] for item in cases_response.get("items", [])]
    except DashboardAPIError as exc:
        render_error(exc)
        return

    if not available_case_ids:
        st.info("No cases exist yet. Seed demo cases first (see docs/UI_DEMO_GUIDE.md).")
        return

    # Default to whichever case the reviewer was last looking at, so
    # "request evidence" on Case Detail flows straight into this page.
    active_case_id = st.session_state.get("selected_case_id")
    default_index = available_case_ids.index(active_case_id) if active_case_id in available_case_ids else 0
    selected_case_id = st.selectbox(
        "Case ID", available_case_ids, index=default_index, key="merchant_response_case_picker"
    )

    try:
        case = client.get_case(selected_case_id)
    except DashboardAPIError as exc:
        render_error(exc)
        return

    st.markdown(f"**Case status:** {case['case_status']}")

    merchant_safe = case.get("merchant_safe_explanation") or {}
    with st.container(border=True):
        st.markdown(f"**Reason category:** {merchant_safe.get('reason_category', '—')}")
        st.markdown(f"**Review signal statement:** {merchant_safe.get('review_signal_statement', '—')}")

    evidence_checklist = case.get("evidence_checklist") or []
    if evidence_checklist:
        st.markdown("**Suggested evidence checklist:**")
        for item in evidence_checklist:
            st.markdown(f"- {item}")

    if case["case_status"] != "EVIDENCE_REQUESTED":
        st.info("Evidence can be submitted only after a reviewer requests it.")
        return

    st.markdown("### Submit a response")
    explanation_text = st.text_area("Merchant explanation", key="merchant_explanation_text")
    st.caption("Example simulated evidence references: " + ", ".join(EXAMPLE_REFERENCES))
    references_text = st.text_input(
        "Evidence references (comma-separated, e.g. invoice_demo_001.pdf, refund_policy_demo_url)",
        key="merchant_evidence_references",
    )

    if st.button("Submit response", type="primary", key="submit_merchant_response"):
        evidence_references = [ref.strip() for ref in references_text.split(",") if ref.strip()]
        try:
            client.submit_evidence(selected_case_id, explanation_text, evidence_references)
            st.success("Evidence submitted. The case has moved to EVIDENCE_SUBMITTED.")
            st.rerun()
        except DashboardAPIError as exc:
            render_error(exc)
