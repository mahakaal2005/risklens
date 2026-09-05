"""Source-level safety tests for the Streamlit dashboard. These scan the
actual dashboard/ source files -- they do not require a running backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DASHBOARD_DIR = Path("dashboard")
DASHBOARD_PY_FILES = sorted(DASHBOARD_DIR.rglob("*.py"))

FORBIDDEN_ACTION_WORDS = [
    "freeze", "ban", "terminate", "hold settlement", "reject payment",
    "process payment", "issue refund", "transfer funds",
]
FORBIDDEN_DATA_STRINGS = ["label_high_loss_next_30d", "latent_state_for_demo_only", "support_ticket_rate"]

APPROVED_ACTION_VALUES = {
    "CLEAR_CASE", "MARK_FALSE_POSITIVE", "REQUEST_EVIDENCE",
    "MARK_OPERATIONAL_ISSUE", "ESCALATE_CASE", "MARK_INCONCLUSIVE", "START_REVIEW",
}

EXTERNAL_URL_MARKERS = ["http://cdn.", "https://cdn.", "fonts.googleapis.com", "fonts.gstatic.com", "openai.com", "anthropic.com", "unpkg.com", "jsdelivr.net"]


def _all_dashboard_source() -> str:
    return "\n".join(f.read_text() for f in DASHBOARD_PY_FILES)


def test_no_prohibited_action_labels_in_source():
    """The one intentional exception is the required global disclaimer
    itself (common.py), which must state in the negative that the tool does
    NOT freeze funds, hold settlements, etc. -- that is safe, required
    language, not an offered action. Every other file must be completely
    clean of these words."""
    import re

    for path in DASHBOARD_PY_FILES:
        source = path.read_text()
        if path.name == "common.py":
            # Strip the GLOBAL_DISCLAIMER assignment itself -- its whole
            # purpose is to state, in the negative, that these actions are
            # NOT available. Every other occurrence anywhere else must
            # still be completely clean.
            source = re.sub(r"GLOBAL_DISCLAIMER = \(.*?\)\n", "", source, flags=re.DOTALL)
        source_lower = source.lower()
        for word in FORBIDDEN_ACTION_WORDS:
            assert word not in source_lower, f"Forbidden action word {word!r} found in {path}"


def test_no_prohibited_data_fields_in_source():
    source = _all_dashboard_source()
    for field in FORBIDDEN_DATA_STRINGS:
        assert field not in source, f"Forbidden field {field!r} found in dashboard source"


def test_merchant_response_page_file_uploader_is_type_restricted():
    """Phase 2 added a real file uploader for evidence attachments (see
    docs/PHASE_2_EVIDENCE_ATTACHMENTS_DESIGN.md) -- this no longer asserts
    zero file upload capability, but it does assert the uploader is
    type-restricted in the UI (defense in depth; the backend is the real
    enforcement point via extension allowlist + magic-byte content check)
    and that no camera/webcam capture exists."""
    source = (DASHBOARD_DIR / "components" / "evidence_form.py").read_text()
    assert "st.file_uploader" in source
    assert "ALLOWED_ATTACHMENT_TYPES" in source
    assert "st.camera_input" not in source


def test_api_client_defaults_to_localhost():
    from dashboard.api_client import DEFAULT_BASE_URL

    assert DEFAULT_BASE_URL.startswith("http://127.0.0.1") or DEFAULT_BASE_URL.startswith("http://localhost")


def test_dashboard_includes_synthetic_data_notice():
    common_source = (DASHBOARD_DIR / "components" / "common.py").read_text()
    assert "synthetic-data demonstration only" in common_source.lower()


def test_no_external_image_cdn_font_or_llm_urls():
    """Every remote URL literal in dashboard source must be one of: the
    local API default, a Render deployment hostname (the dashboard's one
    legitimate cross-service call, see dashboard/api_client.py), or the
    reserved-for-documentation example.org domain used by the illustrative,
    never-sent webhook display in reviewer_actions.py. A real external
    domain (e.g. an actual payment gateway or CDN) appearing here should
    fail this test, not be added to an exception list."""
    source = _all_dashboard_source().lower()
    for marker in EXTERNAL_URL_MARKERS:
        assert marker not in source, f"External resource marker {marker!r} found in dashboard source"

    import re

    urls = re.findall(r"https?://[^\s\"'`]+", _all_dashboard_source())
    for url in urls:
        assert (
            url.startswith("http://127.0.0.1")
            or url.startswith("http://localhost")
            or "onrender.com" in url
            or "example.org" in url
        ), f"Unexpected remote URL in dashboard source: {url}"


def test_simulated_webhook_never_targets_a_real_domain():
    """reviewer_actions.py shows an illustrative webhook payload after a
    reviewer action. This asserts it can never reference a real payment
    domain (only the RFC 2606 reserved example.org), and that the earlier
    fabricated "successfully recorded and synced" success claim is gone --
    the display must be honestly labeled as simulated, not presented as a
    real integration event."""
    source = (DASHBOARD_DIR / "components" / "reviewer_actions.py").read_text()
    assert "razorpay.com" not in source.lower()
    assert "successfully recorded and synced" not in source.lower()
    assert "simulated" in source.lower() or "illustrative" in source.lower()


def test_vertex_ai_call_has_a_labeled_fallback():
    """case_detail.py's optional Gemini call must never present fallback
    text as if it were a live model response -- the reviewer must always
    be able to tell which one they're looking at."""
    source = (DASHBOARD_DIR / "components" / "case_detail.py").read_text()
    assert "fallback_reason" in source
    assert "illustrative example" in source.lower()
    assert "live vertex ai" in source.lower() or "live gemini" in source.lower()


def test_reviewer_action_ui_only_lists_approved_action_values():
    from dashboard.components.reviewer_actions import ACTIONS_BY_STATUS

    for status, actions in ACTIONS_BY_STATUS.items():
        for action_value, _label in actions:
            assert action_value in APPROVED_ACTION_VALUES, f"Unapproved action value {action_value!r} offered in status {status!r}"


def test_resolved_and_escalated_statuses_offer_no_actions():
    from dashboard.components.reviewer_actions import ACTIONS_BY_STATUS

    assert "RESOLVED" not in ACTIONS_BY_STATUS
    assert "ESCALATED" not in ACTIONS_BY_STATUS


def test_dashboard_does_not_import_sqlite_or_sqlalchemy_directly():
    source = _all_dashboard_source()
    assert "import sqlite3" not in source
    assert "import sqlalchemy" not in source
    assert "from sqlalchemy" not in source
    assert "from app.db" not in source
    assert "from app.services" not in source


def test_dashboard_has_exactly_five_pages():
    from dashboard.streamlit_app import PAGES

    assert PAGES == ["Overview", "Review Queue", "Case Detail", "Merchant Response", "Audit Timeline"]


@pytest.mark.parametrize("filename", [f.name for f in DASHBOARD_PY_FILES])
def test_no_undisclosed_llm_or_agent_imports(filename):
    """The one deliberate, disclosed exception is case_detail.py's Vertex
    AI (Gemini) evidence-analysis call -- see its module-level comment and
    dashboard/streamlit_app.py's docstring. Every other file, and every
    other LLM/agent provider, must stay completely clean."""
    path = next(f for f in DASHBOARD_PY_FILES if f.name == filename)
    source = path.read_text().lower()
    for forbidden_import in ["import openai", "import anthropic", "langchain", "import agent"]:
        assert forbidden_import not in source, f"{filename} contains forbidden import: {forbidden_import}"
    if filename != "case_detail.py":
        for forbidden_import in ["import vertexai", "import google.generativeai", "from vertexai"]:
            assert forbidden_import not in source, f"{filename} contains forbidden import: {forbidden_import}"
