"""Login page: shown before any other page when the session is
unauthenticated. Local demo accounts only -- see
docs/PHASE_2_AUTH_DESIGN.md. Not production-grade auth."""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import ClearRiskAPIClient, DashboardAPIError


def render_login(client: ClearRiskAPIClient) -> None:
    st.title("RiskLens")
    st.caption("Sign in to continue. Local demo accounts only -- see scripts/seed_demo_users.py.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if not submitted:
        st.info(
            "No account yet? Run `python3 scripts/seed_demo_users.py` from the repository root -- "
            "it prints demo usernames and passwords to the terminal once.",
            icon="ℹ️",
        )
        return

    if not username or not password:
        st.error("Username and password are both required.")
        return

    try:
        login_response = client.login(username, password)
    except DashboardAPIError as exc:
        st.error(exc.message, icon="🚫")
        return

    st.session_state["session_token"] = login_response["session_token"]
    st.session_state["current_user"] = {
        "role": login_response["role"],
        "actor_id": login_response["actor_id"],
        "display_name": login_response["display_name"],
        "merchant_id": login_response.get("merchant_id"),
    }
    st.rerun()
