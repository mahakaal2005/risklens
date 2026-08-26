import httpx
import pytest

from dashboard.api_client import (
    BASE_URL_ENV_VAR,
    DEFAULT_BASE_URL,
    ClearRiskAPIClient,
    DashboardAPIError,
    get_base_url,
)


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict):
        self.status_code = status_code
        self._json_body = json_body

    def json(self):
        return self._json_body


def test_default_local_base_url_is_used(monkeypatch):
    monkeypatch.delenv(BASE_URL_ENV_VAR, raising=False)
    assert get_base_url() == DEFAULT_BASE_URL
    client = ClearRiskAPIClient()
    assert client.base_url == DEFAULT_BASE_URL


def test_environment_base_url_override_works(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV_VAR, "http://127.0.0.1:9999")
    assert get_base_url() == "http://127.0.0.1:9999"
    client = ClearRiskAPIClient()
    assert client.base_url == "http://127.0.0.1:9999"


def test_network_error_becomes_safe_dashboard_exception(monkeypatch):
    client = ClearRiskAPIClient()

    def raise_connect_error(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "request", raise_connect_error)
    with pytest.raises(DashboardAPIError) as exc_info:
        client.health()
    assert "ConnectError" not in str(exc_info.value)
    assert "Traceback" not in str(exc_info.value)


def test_timeout_becomes_safe_dashboard_exception(monkeypatch):
    client = ClearRiskAPIClient()

    def raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx, "request", raise_timeout)
    with pytest.raises(DashboardAPIError) as exc_info:
        client.health()
    assert "timed out" in exc_info.value.message.lower() or "backend" in exc_info.value.message.lower()


def test_valid_api_response_is_accepted(monkeypatch):
    client = ClearRiskAPIClient()

    def fake_request(method, url, json=None, params=None, timeout=None):
        return _FakeResponse(200, {"status": "ok", "data_mode": "synthetic-only", "payment_actions_enabled": False})

    monkeypatch.setattr(httpx, "request", fake_request)
    body = client.health()
    assert body["status"] == "ok"


def test_missing_synthetic_data_notice_is_rejected(monkeypatch):
    client = ClearRiskAPIClient()

    def fake_request(method, url, json=None, params=None, timeout=None):
        return _FakeResponse(200, {"items": [], "limit": 50, "offset": 0, "total": 0})  # no synthetic_data_notice

    monkeypatch.setattr(httpx, "request", fake_request)
    with pytest.raises(DashboardAPIError):
        client.list_cases()


def test_api_client_does_not_call_external_domains_by_default(monkeypatch):
    called_urls = []

    def fake_request(method, url, json=None, params=None, timeout=None):
        called_urls.append(url)
        return _FakeResponse(200, {"status": "ok"})

    monkeypatch.setattr(httpx, "request", fake_request)
    client = ClearRiskAPIClient()
    client.health()
    for url in called_urls:
        assert url.startswith("http://127.0.0.1:8000") or url.startswith(DEFAULT_BASE_URL)


def test_review_action_sends_only_allowed_fields(monkeypatch):
    captured = {}

    def fake_request(method, url, json=None, params=None, timeout=None):
        captured["json"] = json
        return _FakeResponse(200, {
            "case": {"case_id": "x"}, "new_audit_events": [], "synthetic_data_notice": "Local synthetic-data demonstration only.",
        })

    monkeypatch.setattr(httpx, "request", fake_request)
    client = ClearRiskAPIClient()
    client.submit_review_action("case_1", "REQUEST_EVIDENCE", "analyst_demo_001", "note text")
    assert set(captured["json"].keys()) == {"action", "reviewer_actor_id", "reviewer_note"}


def test_evidence_submission_sends_only_allowed_fields(monkeypatch):
    captured = {}

    def fake_request(method, url, json=None, params=None, timeout=None):
        captured["json"] = json
        return _FakeResponse(200, {
            "evidence_id": "e1", "case_id": "case_1", "case_status": "EVIDENCE_SUBMITTED",
            "submitted_at": "2026-01-01T00:00:00", "evidence_references": [],
            "new_audit_event": {
                "event_sequence_number": 1, "event_timestamp": "2026-01-01T00:00:00",
                "actor_type": "merchant_demo", "actor_id": "merchant_demo_001",
                "event_type": "EVIDENCE_SUBMITTED", "event_payload": {},
            },
            "synthetic_data_notice": "Local synthetic-data demonstration only.",
        })

    monkeypatch.setattr(httpx, "request", fake_request)
    client = ClearRiskAPIClient()
    client.submit_evidence("case_1", "merchant_demo_001", "explanation", ["invoice_demo_001.pdf"])
    assert set(captured["json"].keys()) == {"merchant_actor_id", "merchant_explanation_text", "evidence_references"}


def test_error_response_message_is_extracted_safely(monkeypatch):
    client = ClearRiskAPIClient()

    def fake_request(method, url, json=None, params=None, timeout=None):
        return _FakeResponse(404, {
            "error": {"code": "CASE_NOT_FOUND", "message": "No review case exists for the provided case ID.", "synthetic_data_notice": "Local synthetic-data demonstration only."}
        })

    monkeypatch.setattr(httpx, "request", fake_request)
    with pytest.raises(DashboardAPIError) as exc_info:
        client.get_case("does_not_exist")
    assert exc_info.value.message == "No review case exists for the provided case ID."
