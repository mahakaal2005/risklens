"""ClearRisk Recover — local Streamlit dashboard.

Local synthetic-data demonstration only. This app talks to exactly one
local FastAPI backend (default http://127.0.0.1:8000) over plain HTTP and
nowhere else -- no external API, LLM, CDN asset, or network dependency
exists anywhere in this dashboard. Run with:

    streamlit run dashboard/streamlit_app.py

The FastAPI backend must be started separately:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import ClearRiskAPIClient
from dashboard.components.audit_timeline import render_audit_timeline
from dashboard.components.case_detail import render_case_detail
from dashboard.components.case_list import render_review_queue
from dashboard.components.common import render_connection_status, render_disclaimer, render_page_notice
from dashboard.components.evidence_form import render_merchant_response
from dashboard.components.metrics import render_overview

st.set_page_config(page_title="ClearRisk Recover", page_icon="🛡️", layout="wide")

PAGES = ["Overview", "Review Queue", "Case Detail", "Merchant Response", "Audit Timeline"]


def main() -> None:
    client = ClearRiskAPIClient()

    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Overview"

    with st.sidebar:
        st.title("ClearRisk Recover")
        st.caption("Local synthetic-data demo")
        default_index = PAGES.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in PAGES else 0
        page = st.radio("Navigate", PAGES, index=default_index, key="nav_radio")
        st.session_state["nav_page"] = page

        active_case = st.session_state.get("selected_case_id")
        if active_case:
            st.divider()
            st.caption("Active case")
            st.code(active_case, language=None)

        st.divider()

    # Sidebar carries the persistent safety statement and a quiet backend
    # indicator; the page body stays free for the actual work.
    render_connection_status(client)
    render_disclaimer()

    render_page_notice()

    page = st.session_state["nav_page"]
    if page == "Overview":
        render_overview(client)
    elif page == "Review Queue":
        render_review_queue(client)
    elif page == "Case Detail":
        render_case_detail(client)
    elif page == "Merchant Response":
        render_merchant_response(client)
    elif page == "Audit Timeline":
        render_audit_timeline(client)


if __name__ == "__main__":
    main()
