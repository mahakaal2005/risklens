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
    source = _all_dashboard_source().lower()
    for marker in EXTERNAL_URL_MARKERS:
        assert marker not in source, f"External resource marker {marker!r} found in dashboard source"
    # No remote http(s) URL should appear anywhere except the local API default, Render, or Razorpay webhook simulation.
    import re

    urls = re.findall(r"https?://[^\s\"'`]+", _all_dashboard_source())
    for url in urls:
        assert (
            url.startswith("http://127.0.0.1") 
            or url.startswith("http://localhost") 
            or "onrender.com" in url
            or "api.razorpay.com" in url
            or "{value}" in url
        ), f"Unexpected remote URL in dashboard source: {url}"


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
def test_no_llm_or_agent_imports(filename):
    path = next(f for f in DASHBOARD_PY_FILES if f.name == filename)
    source = path.read_text().lower()
    for forbidden_import in ["import openai", "import anthropic", "langchain", "import agent"]:
        assert forbidden_import not in source, f"{filename} contains forbidden import: {forbidden_import}"
