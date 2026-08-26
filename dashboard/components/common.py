"""Shared UI helpers: safety disclaimer, backend connection status, risk
intensity badges, and safe formatting. No business logic lives here --
only display helpers around data already returned by the API client.
"""

from __future__ import annotations

import html

import streamlit as st

from dashboard.api_client import ClearRiskAPIClient, DashboardAPIError

GLOBAL_DISCLAIMER = (
    "Local synthetic-data demonstration only. This tool does not process payments, "
    "hold settlements, freeze funds, ban merchants, or make final fraud decisions."
)

# Short per-page reminder. The full statement above lives in the sidebar so
# it is always on screen without consuming the top of every page body.
PAGE_NOTICE = "Synthetic demo data · review recommendations only · not a final fraud decision."

BACKEND_START_COMMAND = "uvicorn app.main:app --reload"

INTENSITY_COLORS = {
    "low": "#1a7f37",
    "medium": "#9a6700",
    "high": "#cf222e",
}
INTENSITY_BACKGROUNDS = {
    "low": "#dafbe1",
    "medium": "#fff8c5",
    "high": "#ffebe9",
}


def render_disclaimer() -> None:
    """Full safety statement, rendered once in the sidebar so it stays on
    screen for the whole session without pushing page content down."""
    st.sidebar.warning(GLOBAL_DISCLAIMER, icon="⚠️")


def render_page_notice() -> None:
    """One-line reminder at the top of a page body."""
    st.caption(PAGE_NOTICE)


def render_connection_status(client: ClearRiskAPIClient) -> bool:
    """Quiet in the sidebar when healthy; loud in the page body only when
    broken. A working backend is the expected case and does not deserve a
    full-width alert on all five pages."""
    try:
        health = client.health()
        st.sidebar.caption(f"🟢 Backend connected · {health.get('data_mode', 'unknown')}")
        return True
    except DashboardAPIError as exc:
        st.sidebar.caption("🔴 Backend unavailable")
        st.error(
            f"Backend unavailable at {client.base_url} — start it with:\n\n`{BACKEND_START_COMMAND}`\n\n"
            f"({exc.message})",
            icon="🔌",
        )
        return False


def intensity_badge(intensity: str | None) -> str:
    """Returns an HTML span with color AND text label -- never color alone.
    The label is HTML-escaped: the API types this field as a plain string,
    so nothing upstream structurally guarantees it stays within the known
    low/medium/high set."""
    label = html.escape((intensity or "unknown").strip())
    key = label.lower()
    color = INTENSITY_COLORS.get(key, "#57606a")
    background = INTENSITY_BACKGROUNDS.get(key, "#eaeef2")
    return (
        f'<span style="background-color:{background}; color:{color}; '
        f'padding:2px 8px; border-radius:6px; font-weight:600; font-size:0.85em;">{label}</span>'
    )


def render_intensity_badge(intensity: str | None) -> None:
    st.markdown(intensity_badge(intensity), unsafe_allow_html=True)


def format_timestamp(value: str | None) -> str:
    if not value:
        return "—"
    return str(value).replace("T", " ").split(".")[0]


def safe_get(data: dict, key: str, default="—"):
    value = data.get(key)
    return value if value not in (None, "") else default


def render_error(exc: DashboardAPIError) -> None:
    st.error(exc.message, icon="🚫")
