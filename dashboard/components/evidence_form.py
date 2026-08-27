"""Merchant Response page: simulates a merchant responding to an evidence
request. Evidence references remain validated strings; an optional real
file attachment can also be uploaded (Phase 2), submitted through
POST /cases/{case_id}/evidence/{evidence_id}/attachments. The backend
independently validates the file (extension allowlist, size cap,
magic-byte content check) -- this page does not duplicate that logic, it
only surfaces whatever the backend reports.
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

ALLOWED_ATTACHMENT_TYPES = ["pdf", "txt", "png", "jpg", "jpeg"]


def render_merchant_response(client: ClearRiskAPIClient) -> None:
    st.title("Merchant Response")
    st.caption("Simulates a merchant submitting a response to a review case.")

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
    uploaded_file = st.file_uploader(
        "Optional file attachment", type=ALLOWED_ATTACHMENT_TYPES, key="merchant_evidence_attachment",
    )
    st.caption(f"Allowed types: {', '.join(ALLOWED_ATTACHMENT_TYPES)}. Max size 5 MB.")

    if st.button("Submit response", type="primary", key="submit_merchant_response"):
        evidence_references = [ref.strip() for ref in references_text.split(",") if ref.strip()]
        try:
            submission = client.submit_evidence(selected_case_id, explanation_text, evidence_references)
        except DashboardAPIError as exc:
            render_error(exc)
            return

        if uploaded_file is not None:
            try:
                client.upload_attachment(
                    selected_case_id,
                    submission["evidence_id"],
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            except DashboardAPIError as exc:
                # The text response was already submitted successfully --
                # say so plainly instead of implying the whole thing failed.
                st.warning(f"Evidence text was submitted, but the file attachment was rejected: {exc.message}")
                st.rerun()
                return

        st.success("Evidence submitted. The case has moved to EVIDENCE_SUBMITTED.")
        st.rerun()
