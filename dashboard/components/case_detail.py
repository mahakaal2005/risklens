"""Case Detail page: why a merchant-week was flagged, and the reviewer's
next-step controls. Reads exclusively from GET /cases/{case_id}; renders
only fields the API safely provides. Where the API response does not
contain a value needed for a comparison (e.g. trend charts), this page
says so plainly rather than inferring or fabricating a number.
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import ClearRiskAPIClient, DashboardAPIError
from dashboard.components.common import render_error, render_intensity_badge
from dashboard.components.reviewer_actions import render_reviewer_actions

UNCERTAINTY_TEXT = "This is a review signal, not a final fraud finding."

# Presentation-only label mapping for already-known rule IDs -- not
# fabricated data, just a friendlier display of the same triggered_rules
# list the API already returns.
RULE_LABELS = {
    "REFUND_RATE_SPIKE": "Refund rate spike",
    "CHARGEBACK_RATE_SPIKE": "Chargeback rate spike",
    "EVIDENCE_COVERAGE_GAP": "Evidence coverage gap",
    "SUPPORT_OPERATIONAL_STRESS": "Support operational stress",
    "COMBINED_LOSS_SIGNAL": "Combined loss signal",
}


def _render_case_header(case: dict) -> None:
    """Compact identity strip. Everything a reviewer needs to know who and
    what they are looking at, without a screen of key/value markdown."""
    cols = st.columns(4)
    cols[0].markdown(f"**Merchant**  \n`{case['merchant_id']}`")
    cols[1].markdown(f"**Week start**  \n{case['week_start']}")
    cols[2].markdown(f"**Status**  \n{case['case_status']}")
    with cols[3]:
        st.markdown("**Risk signal**")
        render_intensity_badge(case.get("risk_signal_intensity"))

    st.markdown(f"**Recommended workflow action:** {case['recommendation']}")
    st.caption(case.get("synthetic_data_notice", ""))


def _render_analyst_detail(case: dict) -> None:
    st.caption("Analyst-only. Never shown to the merchant.")
    cols = st.columns(2)
    model_probability = case.get("model_probability")
    cols[0].markdown(
        f"**Model probability**  \n{model_probability:.4f}"
        if model_probability is not None
        else "**Model probability**  \nNot available (rules-only fallback)"
    )
    cols[0].markdown(f"**Rules-only score**  \n{case.get('rules_only_score', '—')}")
    cols[1].markdown(f"**Model version**  \n{case.get('model_version') or '—'}")
    cols[1].markdown(f"**Rules version**  \n{case.get('rules_version') or '—'}")
    st.markdown(f"**Case ID:** `{case['case_id']}`")

    st.caption(
        "Top model factors and current-vs-prior trend comparisons are not available from the "
        "case-detail API response — that endpoint does not expose them yet (a known gap, tracked "
        "separately). Nothing is inferred or filled in here."
    )


def _render_why_flagged(case: dict) -> None:
    st.write(case.get("analyst_summary", ""))

    triggered_rules = case.get("triggered_rules") or []
    if triggered_rules:
        st.markdown("**Triggered rules:**")
        for rule_id in triggered_rules:
            st.markdown(f"- {RULE_LABELS.get(rule_id, rule_id)}")
    else:
        st.caption("No rules triggered for this merchant-week.")

    st.markdown(f"**Policy explanation:** {case.get('policy_explanation', '')}")
    st.info(UNCERTAINTY_TEXT, icon="ℹ️")


def _render_evidence_checklist(case: dict) -> None:
    st.caption("Suggested, not a final demand.")
    checklist = case.get("evidence_checklist") or []
    if not checklist:
        st.caption("No specific evidence was requested for this merchant-week.")
        return
    for item in checklist:
        st.markdown(f"- {item}")


def _render_submitted_evidence(client: ClearRiskAPIClient, case: dict) -> None:
    submissions = case.get("evidence_submissions") or []
    if not submissions:
        st.caption("No evidence has been submitted for this case yet.")
        return

    st.markdown("### Submitted evidence")
    for submission in submissions:
        with st.container(border=True):
            st.markdown(f"**Submitted:** {submission.get('submitted_at', '—')} · **Status:** {submission.get('status', '—')}")
            references = submission.get("evidence_references") or []
            if references:
                st.markdown("**Evidence references:** " + ", ".join(references))

            attachments = submission.get("attachments") or []
            if not attachments:
                st.caption("No file attachments for this submission.")
                continue

            for attachment in attachments:
                cols = st.columns([3, 1])
                size_kb = attachment["size_bytes"] / 1024
                cols[0].markdown(f"📎 {attachment['original_filename']} ({size_kb:.1f} KB)")
                download_key = f"download_{attachment['attachment_id']}"
                if cols[1].button("Download", key=download_key):
                    try:
                        content, content_type = client.download_attachment(
                            case["case_id"], submission["evidence_id"], attachment["attachment_id"],
                        )
                        st.download_button(
                            "Save file",
                            data=content,
                            file_name=attachment["original_filename"],
                            mime=content_type,
                            key=f"save_{attachment['attachment_id']}",
                        )
                    except DashboardAPIError as exc:
                        render_error(exc)


def _render_merchant_safe_preview(case: dict) -> None:
    st.caption("Only the merchant-safe explanation — no probability, score, or internal policy detail.")
    merchant_safe = case.get("merchant_safe_explanation") or {}
    with st.container(border=True):
        st.markdown(f"**Reason category:** {merchant_safe.get('reason_category', '—')}")
        st.markdown(f"**Review signal statement:** {merchant_safe.get('review_signal_statement', '—')}")
        reasons = merchant_safe.get("reasons") or []
        if reasons:
            st.markdown("**Reasons:**")
            for reason in reasons:
                st.markdown(f"- {reason}")
        suggested_evidence = merchant_safe.get("suggested_evidence") or []
        if suggested_evidence:
            st.markdown("**Suggested evidence:**")
            for item in suggested_evidence:
                st.markdown(f"- {item}")
        st.markdown(f"**Appeal placeholder:** {merchant_safe.get('appeal_placeholder', '—')}")


def render_case_detail(client: ClearRiskAPIClient) -> None:
    st.title("Case Detail")

    selected_case_id = st.session_state.get("selected_case_id")

    try:
        cases_response = client.list_cases(limit=100)
        available_case_ids = [item["case_id"] for item in cases_response.get("items", [])]
    except DashboardAPIError as exc:
        render_error(exc)
        return

    if not available_case_ids:
        st.info("No cases exist yet. Seed demo cases first (see docs/UI_DEMO_GUIDE.md).")
        return

    if selected_case_id not in available_case_ids:
        selected_case_id = available_case_ids[0]

    picked = st.selectbox(
        "Case ID",
        available_case_ids,
        index=available_case_ids.index(selected_case_id) if selected_case_id in available_case_ids else 0,
        key="case_detail_case_picker",
    )
    st.session_state["selected_case_id"] = picked

    try:
        case = client.get_case(picked)
    except DashboardAPIError as exc:
        render_error(exc)
        return

    _render_case_header(case)
    st.divider()

    # Reviewer actions sit directly under the header: recording a decision
    # is the reviewer's primary job, so it must not be buried in a tab.
    render_reviewer_actions(client, case)
    st.divider()

    why_tab, evidence_tab, merchant_tab, analyst_tab = st.tabs(
        ["Why flagged", "Evidence checklist", "What the merchant sees", "Analyst detail"]
    )
    with why_tab:
        _render_why_flagged(case)
    with evidence_tab:
        _render_evidence_checklist(case)
        st.divider()
        _render_submitted_evidence(client, case)
    with merchant_tab:
        _render_merchant_safe_preview(case)
    with analyst_tab:
        _render_analyst_detail(case)
