import json

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_db
from app.db.database import create_db_engine, init_db, make_session_factory, session_scope
from app.main import app
from app.services.case_service import create_case_from_packet
from tests.conftest import make_bearer_headers


@pytest.fixture()
def session_factory(tmp_path):
    db_path = tmp_path / "test_api_evidence.db"
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


def _merchant_headers(session_factory, merchant_id):
    return make_bearer_headers(
        session_factory, "merchant", f"merchant_demo_actor_{merchant_id}", "Demo Merchant", merchant_id=merchant_id,
    )


def _case_id_and_merchant_id(client, target_status="REQUEST_EVIDENCE"):
    list_response = client.get("/cases", params={"recommendation": target_status})
    item = list_response.json()["items"][0]
    return item["case_id"], item["merchant_id"]


def _case_id_in_status_requested(client, target_status="REQUEST_EVIDENCE"):
    case_id, merchant_id = _case_id_and_merchant_id(client, target_status)
    client.post(
        f"/cases/{case_id}/review-actions",
        json={"action": "REQUEST_EVIDENCE", "reviewer_note": "Please provide evidence."},
    )
    return case_id, merchant_id


def test_evidence_submission_succeeds_only_in_evidence_requested_status(client, session_factory):
    case_id, merchant_id = _case_id_in_status_requested(client)
    response = client.post(
        f"/cases/{case_id}/evidence",
        json={
            "merchant_explanation_text": "A seasonal sale increased returns.",
            "evidence_references": ["refund_records_demo_001.pdf", "seasonal_sale_summary_demo_001.txt"],
        },
        headers=_merchant_headers(session_factory, merchant_id),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case_status"] == "EVIDENCE_SUBMITTED"
    assert body["evidence_references"] == ["refund_records_demo_001.pdf", "seasonal_sale_summary_demo_001.txt"]


def test_evidence_submission_in_invalid_state_returns_409(client, session_factory):
    case_id, merchant_id = _case_id_and_merchant_id(client, "MANUAL_REVIEW_REQUIRED")  # still OPEN, not EVIDENCE_REQUESTED

    response = client.post(
        f"/cases/{case_id}/evidence",
        json={"merchant_explanation_text": "Explanation.", "evidence_references": []},
        headers=_merchant_headers(session_factory, merchant_id),
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_CASE_TRANSITION"


@pytest.mark.parametrize("bad_reference", ["../../etc/passwd", "http://example.com/file.pdf", "rm -rf /"])
def test_invalid_evidence_reference_returns_422(client, session_factory, bad_reference):
    case_id, merchant_id = _case_id_in_status_requested(client)
    response = client.post(
        f"/cases/{case_id}/evidence",
        json={"merchant_explanation_text": "Explanation.", "evidence_references": [bad_reference]},
        headers=_merchant_headers(session_factory, merchant_id),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_EVIDENCE_REFERENCE"


def test_evidence_submission_unknown_case_returns_404(client, session_factory):
    response = client.post(
        "/cases/does_not_exist/evidence",
        json={"merchant_explanation_text": "Explanation.", "evidence_references": []},
        headers=_merchant_headers(session_factory, "merchant_demo_9999"),
    )
    assert response.status_code == 404


def test_evidence_submission_blank_explanation_returns_422(client, session_factory):
    case_id, merchant_id = _case_id_in_status_requested(client)
    response = client.post(
        f"/cases/{case_id}/evidence",
        json={"merchant_explanation_text": "   ", "evidence_references": []},
        headers=_merchant_headers(session_factory, merchant_id),
    )
    assert response.status_code == 422


def test_merchant_cannot_submit_evidence_for_another_merchants_case(client, session_factory):
    """A merchant-role session scoped to a different merchant_id must not
    be able to submit evidence for this case -- see
    docs/PHASE_2_AUTH_DESIGN.md Section 5."""
    case_id, merchant_id = _case_id_in_status_requested(client)
    wrong_merchant_headers = _merchant_headers(session_factory, merchant_id + "_someone_else")
    response = client.post(
        f"/cases/{case_id}/evidence",
        json={"merchant_explanation_text": "Explanation.", "evidence_references": []},
        headers=wrong_merchant_headers,
    )
    assert response.status_code == 404


def test_reviewer_role_cannot_submit_evidence(client):
    """Evidence submission is a merchant-only action -- a reviewer session
    must be rejected with 403, not silently allowed."""
    case_id, _ = _case_id_in_status_requested(client)
    response = client.post(
        f"/cases/{case_id}/evidence",
        json={"merchant_explanation_text": "Explanation.", "evidence_references": []},
    )  # client's default headers are the reviewer session from the `client` fixture
    assert response.status_code == 403
