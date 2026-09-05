"""RiskLens — local Streamlit dashboard.

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
from dashboard.components.login import render_login
from dashboard.components.metrics import render_overview

st.set_page_config(page_title="RiskLens", page_icon="🛡️", layout="wide")

PAGES = ["Overview", "Review Queue", "Case Detail", "Merchant Response", "Audit Timeline"]

# Which of the 5 pages each authenticated role may navigate to -- a display
# convenience only, mirroring the backend's own role enforcement (Phase 2
# auth). The API is the real authority: an out-of-scope request still gets
# safely rejected server-side with a 401/403/404, never silently allowed
# because a page happened to be reachable. See docs/PHASE_2_AUTH_DESIGN.md.
PAGES_BY_ROLE = {
    "reviewer": ["Overview", "Review Queue", "Case Detail", "Audit Timeline"],
    "merchant": ["Merchant Response", "Case Detail", "Audit Timeline"],
    "risk_manager": ["Overview", "Audit Timeline"],
}


def _render_authenticated_app(client: ClearRiskAPIClient) -> None:
    current_user = st.session_state["current_user"]
    allowed_pages = PAGES_BY_ROLE.get(current_user["role"], PAGES)

    if "nav_page" not in st.session_state or st.session_state["nav_page"] not in allowed_pages:
        st.session_state["nav_page"] = allowed_pages[0]

    with st.sidebar:
        st.title("RiskLens")
        st.caption(f"{current_user['display_name']} · {current_user['role']}")
        default_index = allowed_pages.index(st.session_state["nav_page"])
        page = st.radio("Navigate", allowed_pages, index=default_index, key="nav_radio")
        st.session_state["nav_page"] = page

        active_case = st.session_state.get("selected_case_id")
        if active_case:
            st.divider()
            st.caption("Active case")
            st.code(active_case, language=None)

        st.divider()
        if st.button("Sign out"):
            client.logout()
            st.session_state.pop("session_token", None)
            st.session_state.pop("current_user", None)
            st.rerun()

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


def main() -> None:
    session_token = st.session_state.get("session_token")
    client = ClearRiskAPIClient(session_token=session_token)

    if not session_token or "current_user" not in st.session_state:
        render_disclaimer()
        render_connection_status(client)
        render_login(client)
        return

    _render_authenticated_app(client)


if __name__ == "__main__":
    main()
