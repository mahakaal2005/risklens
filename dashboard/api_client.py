"""HTTP client for the local RiskLens FastAPI backend.

Local synthetic-data demonstration only. This module makes HTTP calls to
exactly one configurable base URL (default http://127.0.0.1:8000) and
nowhere else -- no external API, LLM, or network dependency exists here.
All request/response handling lives in this module so display code
(dashboard/streamlit_app.py, dashboard/components/*.py) never talks to
the network directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 5.0
BASE_URL_ENV_VAR = "CLEARRISK_API_BASE_URL"

SYNTHETIC_DATA_NOTICE_FIELD = "synthetic_data_notice"


class DashboardAPIError(Exception):
    """A safe, display-ready error. Never wraps a raw traceback or
    backend-internal detail -- only a short, user-facing message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


@dataclass
class ReviewActionResult:
    case: dict
    new_audit_events: list[dict]


def get_base_url() -> str:
    return os.environ.get(BASE_URL_ENV_VAR, DEFAULT_BASE_URL)


class ClearRiskAPIClient:
    """Thin wrapper around the FastAPI backend. Every method returns a
    plain dict (already-validated JSON) or raises DashboardAPIError."""

    def __init__(self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS, session_token: str | None = None):
        self.base_url = (base_url or get_base_url()).rstrip("/")
        self.timeout = timeout
        self.session_token = session_token

    def _auth_headers(self) -> dict[str, str] | None:
        return {"Authorization": f"Bearer {self.session_token}"} if self.session_token else None

    def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Makes the actual httpx call and translates network-level
        exceptions into DashboardAPIError. Shared by every method that
        talks to the network, whether the response body ends up being
        JSON (_request) or raw bytes (upload/download_attachment)."""
        try:
            return httpx.request(method, url, timeout=self.timeout, **kwargs)
        except httpx.TimeoutException as exc:
            raise DashboardAPIError(f"The local backend at {self.base_url} timed out. Is it running?") from exc
        except httpx.ConnectError as exc:
            raise DashboardAPIError(f"Could not connect to the local backend at {self.base_url}.") from exc
        except httpx.HTTPError as exc:
            raise DashboardAPIError("A network error occurred while contacting the local backend.") from exc

    def _request(self, method: str, path: str, json_body: dict | None = None, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        response = self._send(method, url, json=json_body, params=params, headers=self._auth_headers())

        try:
            body = response.json()
        except ValueError as exc:
            raise DashboardAPIError("The local backend returned a response that could not be read.") from exc

        if response.status_code >= 400:
            message = self._extract_error_message(body)
            raise DashboardAPIError(message)

        return body

    @staticmethod
    def _extract_error_message(body: Any) -> str:
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
        return "The local backend rejected the request."

    def _require_notice(self, body: dict) -> dict:
        if SYNTHETIC_DATA_NOTICE_FIELD not in body or not body[SYNTHETIC_DATA_NOTICE_FIELD]:
            raise DashboardAPIError("The local backend response was missing the required synthetic-data notice.")
        return body

    # -- Read endpoints --------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def login(self, username: str, password: str) -> dict:
        """Logs in and stores the returned session token on this client
        instance so every subsequent call is authenticated. Returns the
        full login response (role, actor_id, display_name, etc.) for the
        caller to store in Streamlit session state."""
        body = self._require_notice(self._request("POST", "/auth/login", json_body={"username": username, "password": password}))
        self.session_token = body["session_token"]
        return body

    def logout(self) -> None:
        if self.session_token:
            try:
                self._request("POST", "/auth/logout")
            except DashboardAPIError:
                pass  # best-effort -- the session will also just expire on its own
        self.session_token = None

    def get_metrics(self) -> dict:
        # MetricsResponse always includes synthetic_data_notice, even in the
        # not_available case, so this check applies unconditionally.
        return self._require_notice(self._request("GET", "/metrics"))

    def list_cases(
        self,
        status: str | None = None,
        recommendation: str | None = None,
        intensity: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if recommendation:
            params["recommendation"] = recommendation
        if intensity:
            params["intensity"] = intensity
        return self._require_notice(self._request("GET", "/cases", params=params))

    def get_case(self, case_id: str) -> dict:
        return self._require_notice(self._request("GET", f"/cases/{case_id}"))

    def get_audit_events(self, case_id: str) -> dict:
        return self._require_notice(self._request("GET", f"/cases/{case_id}/audit-events"))

    # -- Write endpoints ---------------------------------------------------

    def submit_review_action(self, case_id: str, action: str, reviewer_note: str) -> dict:
        """No actor-id parameter: the reviewer's identity is derived
        server-side from this client's session token, never sent by the
        caller -- see docs/PHASE_2_AUTH_DESIGN.md Section 6."""
        body = {
            "action": action,
            "reviewer_note": reviewer_note,
        }
        return self._require_notice(self._request("POST", f"/cases/{case_id}/review-actions", json_body=body))

    def submit_evidence(
        self,
        case_id: str,
        merchant_explanation_text: str,
        evidence_references: list[str],
    ) -> dict:
        """No actor-id parameter: the merchant's identity is derived
        server-side from this client's session token, never sent by the
        caller -- see docs/PHASE_2_AUTH_DESIGN.md Section 6."""
        body = {
            "merchant_explanation_text": merchant_explanation_text,
            "evidence_references": evidence_references,
        }
        return self._require_notice(self._request("POST", f"/cases/{case_id}/evidence", json_body=body))

    def upload_attachment(
        self,
        case_id: str,
        evidence_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> dict:
        """Real file upload (Phase 2) -- multipart, not JSON, so this method
        bypasses _request() and builds its own httpx call. The backend
        re-validates the file independently (extension allowlist, size cap,
        magic-byte content check); this client does not duplicate that
        validation, only surfaces whatever error the backend returns."""
        url = f"{self.base_url}/cases/{case_id}/evidence/{evidence_id}/attachments"
        response = self._send(
            "POST", url, files={"file": (filename, content, content_type)}, headers=self._auth_headers(),
        )

        try:
            body = response.json()
        except ValueError as exc:
            raise DashboardAPIError("The local backend returned a response that could not be read.") from exc

        if response.status_code >= 400:
            raise DashboardAPIError(self._extract_error_message(body))
        return self._require_notice(body)

    def download_attachment(self, case_id: str, evidence_id: str, attachment_id: str) -> tuple[bytes, str]:
        """Returns (raw file bytes, content_type). Unlike every other method
        here, the response body is not JSON, so this bypasses _request()
        entirely."""
        url = f"{self.base_url}/cases/{case_id}/evidence/{evidence_id}/attachments/{attachment_id}"
        response = self._send("GET", url, headers=self._auth_headers())

        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                raise DashboardAPIError("The local backend rejected the download request.")
            raise DashboardAPIError(self._extract_error_message(body))

        return response.content, response.headers.get("content-type", "application/octet-stream")
