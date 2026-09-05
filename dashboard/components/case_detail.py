"""Case Detail page: why a merchant-week was flagged, and the reviewer's
next-step controls. Reads exclusively from GET /cases/{case_id}; renders
only fields the API safely provides. Where the API response does not
contain a value needed for a comparison (e.g. trend charts), this page
says so plainly rather than inferring or fabricating a number.
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import ClearRiskAPIClient, DashboardAPIError
from dashboard.components.common import get_available_case_ids, render_error, render_intensity_badge, sla_display
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
    with st.container(border=True):
        cols = st.columns(4)
        cols[0].markdown(f"**Merchant**  \n`{case['merchant_id']}`")
        cols[1].markdown(f"**Week start**  \n{case['week_start']}")
        cols[2].markdown(f"**Status**  \n{case['case_status']}")
        with cols[3]:
            st.markdown("**Risk signal**")
            render_intensity_badge(case.get("risk_signal_intensity"))

        st.markdown(f"**Recommended workflow action:** {case['recommendation']}")

        sla_text = sla_display(case)
        if case.get("sla_breached"):
            st.warning(
                f"Review SLA: {sla_text} (simulated in-app indicator only — no real email/SMS notification exists).",
                icon="⏰",
            )
        elif sla_text != "N/A":
            st.caption(f"Review SLA: {sla_text}")

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
        "Top model factors (ranked feature-level contributions) and current-vs-prior trend values "
        "for features that did NOT trigger a rule are not available from the case-detail API response "
        "yet (a known gap, tracked separately). The 'Why flagged' tab does show a concrete before/after "
        "value for each rule that DID trigger. Nothing is inferred or filled in here."
    )


def _render_why_flagged(case: dict) -> None:
    st.write(case.get("analyst_summary", ""))

    explanations_by_rule = {e["rule_id"]: e["explanation"] for e in (case.get("triggered_rule_explanations") or [])}
    triggered_rules = case.get("triggered_rules") or []
    if triggered_rules:
        st.markdown("**Triggered rules:**")
        for rule_id in triggered_rules:
            label = RULE_LABELS.get(rule_id, rule_id)
            concrete_explanation = explanations_by_rule.get(rule_id)
            if concrete_explanation:
                # e.g. "Refund rate spike -- Refund rate increased from 1.63% to 6.45% (4.82% change)."
                st.markdown(f"- **{label}** — {concrete_explanation}")
            else:
                st.markdown(f"- {label}")
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


import logging
import os

_logger = logging.getLogger(__name__)


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
            if attachments:
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
            else:
                st.caption("No file attachments for this submission.")

        # Optional AI-assisted evidence summary. Calls Vertex AI (Gemini)
        # only when GCP_PROJECT_ID and GCP_CREDENTIALS_JSON are configured;
        # otherwise shows a clearly-labeled illustrative example so a
        # reviewer can never mistake canned text for a real model response.
        # This is the one genuine external network dependency in this
        # dashboard -- see dashboard/streamlit_app.py's module docstring.
        st.markdown("### ✨ AI Evidence Analysis")
        with st.spinner("Analyzing merchant evidence using Vertex AI (Gemini 1.5 Flash)..."):
            ai_text = None
            fallback_reason = None
            try:
                import json

                import vertexai
                from google.oauth2 import service_account
                from vertexai.generative_models import GenerativeModel

                gcp_project = os.environ.get("GCP_PROJECT_ID")
                gcp_creds_json = os.environ.get("GCP_CREDENTIALS_JSON")

                if not (gcp_project and gcp_creds_json):
                    fallback_reason = "GCP_PROJECT_ID / GCP_CREDENTIALS_JSON not configured"
                else:
                    creds_dict = json.loads(gcp_creds_json)
                    credentials = service_account.Credentials.from_service_account_info(creds_dict)

                    vertexai.init(project=gcp_project, location="us-central1", credentials=credentials)
                    model = GenerativeModel("gemini-1.5-flash-001")

                    evidence_text = ", ".join(references) if references else "No written evidence provided."
                    prompt = f"""
                    Analyze this merchant's submitted evidence for a risk review.
                    Case details: Merchant ID {case.get('merchant_id')}, Risk Signal: {case.get('risk_signal_intensity')}.
                    Triggered rules: {case.get('triggered_rules')}.
                    Merchant's submitted explanation/evidence: "{evidence_text}".

                    Provide a highly concise, 3-sentence risk assessment for a human analyst.
                    Format your response with these exact bold headings:
                    **Summary:** (1 sentence)
                    **Risk Sentiment:** (Low/Medium/High, 1 sentence)
                    **Recommendation:** (1 sentence)
                    """

                    response = model.generate_content(prompt)
                    ai_text = response.text
            except Exception as exc:
                # Logged, not swallowed -- a reviewer sees the fallback
                # label below, and the real cause is still visible in
                # server logs for debugging (never a raw traceback in the UI).
                _logger.warning("Vertex AI evidence analysis failed, using fallback example: %s", exc)
                fallback_reason = f"live call failed ({type(exc).__name__})"

            if ai_text is not None:
                st.caption("✨ Live Vertex AI (Gemini 1.5 Flash) response.")
                st.info(ai_text)
            else:
                st.caption(f"⚠️ Illustrative example only -- {fallback_reason}. This is not a real model response.")
                st.info(
                    "**Summary:** The merchant's explanation strongly correlates with the timeframe of the observed risk spike. "
                    "The provided shipping context matches known logistical delays for this region.\n\n"
                    "**Risk Sentiment:** **Low**. The evidence appears credible and adequately explains the temporary increase in refund rates.\n\n"
                    "**Recommendation:** Accept the provided evidence and resolve the case as a False Positive."
                )



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

    available_case_ids = get_available_case_ids(client)
    if available_case_ids is None:
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
