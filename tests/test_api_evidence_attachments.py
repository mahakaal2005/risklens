"""Tests for POST/GET .../evidence/{evidence_id}/attachments -- real file
upload and download for merchant evidence (Phase 2)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_db
from app.db.database import create_db_engine, init_db, make_session_factory, session_scope
from app.main import app
from app.services.case_service import create_case_from_packet
from tests.conftest import make_bearer_headers

PDF_BYTES = b"%PDF-1.4\n%fake pdf content for tests"


@pytest.fixture()
def session_factory(tmp_path, monkeypatch):
    # Isolate uploaded-file storage per test too, not just the database.
    from app.services import evidence_attachment_service

    monkeypatch.setattr(evidence_attachment_service, "STORAGE_DIR", tmp_path / "evidence_attachments")

    db_path = tmp_path / "test_api_evidence_attachments.db"
    engine = create_db_engine(f"sqlite:///{db_path}")
    init_db(engine)
    factory = make_session_factory(engine)

    def override_get_db():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    with open("demo_data/demo_case_packets.json", "r", encoding="utf-8") as f:
        packets = json.load(f)
    with session_scope(factory) as session:
        for packet in packets.values():
            create_case_from_packet(session, packet)

    yield factory
    app.dependency_overrides.clear()


@pytest.fixture()
def client(session_factory):
    reviewer_headers = make_bearer_headers(session_factory, "reviewer", "analyst_demo_001", "Demo Reviewer")
    test_client = TestClient(app)
    test_client.headers.update(reviewer_headers)
    return test_client


def _merchant_headers(session_factory, merchant_id, suffix="1"):
    return make_bearer_headers(
        session_factory, "merchant", f"merchant_demo_actor_{merchant_id}_{suffix}", "Demo Merchant",
        merchant_id=merchant_id, username=f"merchant_{merchant_id}_{suffix}",
    )


def _submit_evidence_and_get_id(client, session_factory):
    list_response = client.get("/cases", params={"recommendation": "REQUEST_EVIDENCE"})
    item = list_response.json()["items"][0]
    case_id, merchant_id = item["case_id"], item["merchant_id"]

    client.post(
        f"/cases/{case_id}/review-actions",
        json={"action": "REQUEST_EVIDENCE", "reviewer_note": "Please provide evidence."},
    )
    merchant_headers = _merchant_headers(session_factory, merchant_id)
    submit_response = client.post(
        f"/cases/{case_id}/evidence",
        json={"merchant_explanation_text": "Explanation.", "evidence_references": []},
        headers=merchant_headers,
    )
    evidence_id = submit_response.json()["evidence_id"]
    return case_id, evidence_id, merchant_id, merchant_headers


def test_merchant_can_upload_a_valid_pdf_attachment(client, session_factory):
    case_id, evidence_id, _merchant_id, merchant_headers = _submit_evidence_and_get_id(client, session_factory)

    response = client.post(
        f"/cases/{case_id}/evidence/{evidence_id}/attachments",
        files={"file": ("invoice.pdf", PDF_BYTES, "application/pdf")},
        headers=merchant_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["attachment"]["original_filename"] == "invoice.pdf"
    assert body["attachment"]["content_type"] == "application/pdf"
    assert body["attachment"]["size_bytes"] == len(PDF_BYTES)
    assert body["new_audit_event"]["event_type"] == "EVIDENCE_ATTACHMENT_UPLOADED"


def test_disallowed_file_type_is_rejected(client, session_factory):
    case_id, evidence_id, _merchant_id, merchant_headers = _submit_evidence_and_get_id(client, session_factory)

    response = client.post(
        f"/cases/{case_id}/evidence/{evidence_id}/attachments",
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
        headers=merchant_headers,
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_EVIDENCE_REFERENCE"


def test_content_mismatched_with_extension_is_rejected(client, session_factory):
    case_id, evidence_id, _merchant_id, merchant_headers = _submit_evidence_and_get_id(client, session_factory)

    fake_pdf = b"MZ\x90\x00\x03\x00\x00\x00"
    response = client.post(
        f"/cases/{case_id}/evidence/{evidence_id}/attachments",
        files={"file": ("totally_a_pdf.pdf", fake_pdf, "application/pdf")},
        headers=merchant_headers,
    )
    assert response.status_code == 422


def test_reviewer_role_cannot_upload_attachments(client, session_factory):
    case_id, evidence_id, _merchant_id, _merchant_headers = _submit_evidence_and_get_id(client, session_factory)

    response = client.post(
        f"/cases/{case_id}/evidence/{evidence_id}/attachments",
        files={"file": ("invoice.pdf", PDF_BYTES, "application/pdf")},
    )  # client's default headers are the reviewer session
    assert response.status_code == 403


def test_merchant_cannot_upload_to_another_merchants_case(client, session_factory):
    case_id, evidence_id, merchant_id, _owner_headers = _submit_evidence_and_get_id(client, session_factory)
    wrong_merchant_headers = _merchant_headers(session_factory, merchant_id + "_someone_else", suffix="2")

    response = client.post(
        f"/cases/{case_id}/evidence/{evidence_id}/attachments",
        files={"file": ("invoice.pdf", PDF_BYTES, "application/pdf")},
        headers=wrong_merchant_headers,
    )
    assert response.status_code == 404


def test_reviewer_can_download_an_uploaded_attachment(client, session_factory):
    case_id, evidence_id, _merchant_id, merchant_headers = _submit_evidence_and_get_id(client, session_factory)
    upload_response = client.post(
        f"/cases/{case_id}/evidence/{evidence_id}/attachments",
        files={"file": ("invoice.pdf", PDF_BYTES, "application/pdf")},
        headers=merchant_headers,
    )
    attachment_id = upload_response.json()["attachment"]["attachment_id"]

    download_response = client.get(f"/cases/{case_id}/evidence/{evidence_id}/attachments/{attachment_id}")
    assert download_response.status_code == 200
    assert download_response.content == PDF_BYTES
    assert "invoice.pdf" in download_response.headers["content-disposition"]


def test_merchant_cannot_download_another_merchants_attachment(client, session_factory):
    case_id, evidence_id, merchant_id, merchant_headers = _submit_evidence_and_get_id(client, session_factory)
    upload_response = client.post(
        f"/cases/{case_id}/evidence/{evidence_id}/attachments",
        files={"file": ("invoice.pdf", PDF_BYTES, "application/pdf")},
        headers=merchant_headers,
    )
    attachment_id = upload_response.json()["attachment"]["attachment_id"]

    wrong_merchant_headers = _merchant_headers(session_factory, merchant_id + "_someone_else", suffix="2")
    response = client.get(
        f"/cases/{case_id}/evidence/{evidence_id}/attachments/{attachment_id}", headers=wrong_merchant_headers,
    )
    assert response.status_code == 404


def test_unknown_attachment_id_returns_404(client, session_factory):
    case_id, evidence_id, _merchant_id, _merchant_headers = _submit_evidence_and_get_id(client, session_factory)
    response = client.get(f"/cases/{case_id}/evidence/{evidence_id}/attachments/does_not_exist")
    assert response.status_code == 404


def test_attachment_appears_in_case_detail_evidence_submissions(client, session_factory):
    case_id, evidence_id, _merchant_id, merchant_headers = _submit_evidence_and_get_id(client, session_factory)
    client.post(
        f"/cases/{case_id}/evidence/{evidence_id}/attachments",
        files={"file": ("invoice.pdf", PDF_BYTES, "application/pdf")},
        headers=merchant_headers,
    )

    case_detail = client.get(f"/cases/{case_id}").json()
    submission = next(e for e in case_detail["evidence_submissions"] if e["evidence_id"] == evidence_id)
    assert len(submission["attachments"]) == 1
    assert submission["attachments"][0]["original_filename"] == "invoice.pdf"
