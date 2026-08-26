import json

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_db
from app.db.database import create_db_engine, init_db, make_session_factory, session_scope
from app.main import app
from app.services.case_service import create_case_from_packet


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test_api_evidence.db"
    engine = create_db_engine(f"sqlite:///{db_path}")
    init_db(engine)
    session_factory = make_session_factory(engine)

    def override_get_db():
        session = session_factory()
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
    with session_scope(session_factory) as session:
        for packet in packets.values():
            create_case_from_packet(session, packet)

    yield TestClient(app)
    app.dependency_overrides.clear()


def _case_id_in_status_requested(client, target_status="REQUEST_EVIDENCE"):
    list_response = client.get("/cases", params={"recommendation": target_status})
    case_id = list_response.json()["items"][0]["case_id"]
    client.post(
        f"/cases/{case_id}/review-actions",
        json={"action": "REQUEST_EVIDENCE", "reviewer_actor_id": "analyst_demo_001", "reviewer_note": "Please provide evidence."},
    )
    return case_id


def test_evidence_submission_succeeds_only_in_evidence_requested_status(client):
    case_id = _case_id_in_status_requested(client)
    response = client.post(
        f"/cases/{case_id}/evidence",
        json={
            "merchant_actor_id": "merchant_demo_001",
            "merchant_explanation_text": "A seasonal sale increased returns.",
            "evidence_references": ["refund_records_demo_001.pdf", "seasonal_sale_summary_demo_001.txt"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case_status"] == "EVIDENCE_SUBMITTED"
    assert body["evidence_references"] == ["refund_records_demo_001.pdf", "seasonal_sale_summary_demo_001.txt"]


def test_evidence_submission_in_invalid_state_returns_409(client):
    list_response = client.get("/cases", params={"recommendation": "MANUAL_REVIEW_REQUIRED"})
    case_id = list_response.json()["items"][0]["case_id"]  # still OPEN, not EVIDENCE_REQUESTED

    response = client.post(
        f"/cases/{case_id}/evidence",
        json={"merchant_actor_id": "merchant_demo_001", "merchant_explanation_text": "Explanation.", "evidence_references": []},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_CASE_TRANSITION"


@pytest.mark.parametrize("bad_reference", ["../../etc/passwd", "http://example.com/file.pdf", "rm -rf /"])
def test_invalid_evidence_reference_returns_422(client, bad_reference):
    case_id = _case_id_in_status_requested(client)
    response = client.post(
        f"/cases/{case_id}/evidence",
        json={"merchant_actor_id": "merchant_demo_001", "merchant_explanation_text": "Explanation.", "evidence_references": [bad_reference]},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_EVIDENCE_REFERENCE"


def test_evidence_submission_unknown_case_returns_404(client):
    response = client.post(
        "/cases/does_not_exist/evidence",
        json={"merchant_actor_id": "merchant_demo_001", "merchant_explanation_text": "Explanation.", "evidence_references": []},
    )
    assert response.status_code == 404


def test_evidence_submission_blank_explanation_returns_422(client):
    case_id = _case_id_in_status_requested(client)
    response = client.post(
        f"/cases/{case_id}/evidence",
        json={"merchant_actor_id": "merchant_demo_001", "merchant_explanation_text": "   ", "evidence_references": []},
    )
    assert response.status_code == 422
